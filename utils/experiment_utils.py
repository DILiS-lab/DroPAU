from models.prob_nn import ProbabilisticFeedForwardNetwork
import sys
import os

sys.path.append("../CLUE")
import warnings
from typing import Optional
import pandas as pd
import shap
import numpy as np
import torch
from timeit import default_timer as timer

from .infoshap_xgboost import train_xgboost, train_xgboost_var, infoboost_explain
from CLUE.VAE.fc_gauss import VAE_gauss_net
from CLUE.VAE.train import train_VAE
from CLUE.interpret.CLUE import CLUE
from CLUE.interpret.visualization_tools import latent_project_gauss
from CLUE.src.utils import Datafeed, Ln_distance


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*No audio backend is available.*")


def linear_model(
    n_samples: int, n_features: int, noise: float = 0.05, random_state: int = 1
):
    """
    Generate random samples and labels for a linear regression model.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - n_features (int): Number of features.
    - noise (float, optional): Standard deviation of Gaussian noise added to the labels (default is 0.05).
    - random_state (int, optional): Seed for random number generation (default is 1).

    Returns:
    - features (numpy.ndarray): Generated random feature matrix of shape (n_samples, n_features).
    - y (numpy.ndarray): Generated labels using a linear combination of features and weights with added noise.

    """

    np.random.seed(random_state)

    # Generate random samples and coefficients
    features = np.random.randn(n_samples, n_features)
    weights = np.random.uniform(-1, 1, n_features)
    bias = np.random.uniform(-1, 1, n_features)  # TODO bias not used
    # Compute y using linear combination of X and coef
    y = np.dot(features, weights) + noise * np.random.randn(n_samples)

    return features, y


def noise_model(n: int, k: int, noise_level: float):
    """
    Generate uncertainty features and standard deviation for the heteroscedastic noise.

    Parameters:
    - n (int): Number of samples to generate.
    - k (int): Number of features.
    - noise_level (float): Standard deviation of the error of the model which generates the heteroscedastic noise.

    Returns:
    - features (numpy.ndarray): Generated random feature matrix of shape (n, k).
    - std (numpy.ndarray): Generated standard deviation of the heteroscedastic noise of shape (n,).

    Example:
    features, labels = noise_model(n=100, k=3, noise_level=0.1)
    """
    # Generate random noise
    noise = np.random.normal(0, noise_level, n)

    # Generate random weights for linear and quadratic terms
    bias = np.random.uniform(0.5, 1, 1) * np.random.choice([-1, 1], 1)
    weights_linear = np.random.uniform(0.5, 1, k) * np.random.choice([-1, 1], k)
    weights_quadratic = np.random.uniform(0.5, 1, k) * np.random.choice([-1, 1], k)

    # Generate random weights for interaction terms
    weights_interaction = np.random.uniform(
        0.5, 1, int(k * (k - 1) / 2)
    ) * np.random.choice([-1, 1], int(k * (k - 1) / 2))

    # Generate random features
    features = np.random.normal(0, 1, (n, k))

    # Compute linear and quadratic contributions
    labels = (
        np.dot(features, weights_linear)
        + np.dot(features**2, weights_quadratic)
        + bias
    )

    # Compute interaction terms contributions using broadcasting
    i, j = np.triu_indices(k, 1)  # Get upper triangular indices, excluding the diagonal
    interaction_terms = features[:, i] * features[:, j]
    labels += interaction_terms @ weights_interaction

    # Add noise
    labels += noise

    stds = np.abs(labels)

    return features, stds


def simple_noise_model(n: int, k: int, noise_level: float):
    features = np.random.normal(0, 1, (n, k))
    stds = noise_level * np.abs(np.sum(features, axis=1))
    return features, stds


def get_data(
    remake_data: bool,
    n_train: int,
    n_test: int,
    noise_scaler: float,
    save_dir: str,
    identifier: str,
    k_mean: int = 70,
    k_noise: int = 5,
    k_mixed: int = 0,
    random_state: int = 1,
):
    """
    Generate or load synthetic dataset for training, validation, and testing of an uncertainty aware model.

    Parameters:
    - remake_data (bool): If True, regenerate the dataset. If False, load the dataset from a file.
    - n_train (int): Number of training samples (includes validation samples, as we use validation for early stopping)
    - n_test (int): Number of testing samples.
    - noise_scaler (float): Scaling factor for the heteroscedastic noise in the dataset.
    - save_dir (str): Directory where the dataset should be saved or loaded from.
    - identifier (str): Unique identifier for the dataset file.
    - k_mean (int, optional): Number of features for the linear model (default is 70).
    - k_noise (int, optional): Number of features for the noise model (default is 5).
    - k_mixed (int, optional): Number of features for the mixed model (default is 5).
    - random_state (int, optional): Seed for random number generation (default is 1).

    Returns:
    - x_train (torch.Tensor): Training features as a torch tensor.
    - x_val (torch.Tensor): Validation features as a torch tensor.
    - x_test (torch.Tensor): Testing features as a torch tensor.
    - y_train (torch.Tensor): Training labels as a torch tensor.
    - y_val (torch.Tensor): Validation labels as a torch tensor.
    - y_test (torch.Tensor): Testing labels as a torch tensor.
    - y_means (numpy.ndarray): Mean values used for scaling the labels.
    - y_stds (numpy.ndarray): Standard deviation values used for scaling the labels.
    - feature_names (list of str): Names of the features in the dataset.

    """
    n = n_train + n_test
    if remake_data:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        model_error_linear = 0.02
        model_error_noise = 0.05

        # generate the mean features and mean labels
        data = linear_model(
            n, k_mean, noise=model_error_linear, random_state=random_state
        )
        inputs = data[0]
        output = data[1]
        data_noise = noise_model(n, k_noise, model_error_noise)

        # generate the heteroscedastic noise based on output of the noise model
        noise = np.random.normal(loc=0.0, scale=noise_scaler * data_noise[1], size=n)
        output = output + noise
        feature_names = [f"feature_{i}" for i in range(inputs.shape[1])] + [
            f"noise_feature_{i}" for i in range(data_noise[0].shape[1])
        ]
        inputs = np.concatenate((inputs, data_noise[0]), axis=1)


        if k_mixed > 0:
            data_mixed = noise_model(
                n, k_mixed, model_error_noise
            )
            # standardize the data
            location_change = (data_mixed[1]-np.mean(data_mixed[1]))/np.std(data_mixed[1])
            # scale to the same range as the original data for meaningful contribution
            shift = np.random.normal(loc=location_change*1.2, scale=noise_scaler*data_mixed[1], size=n) 
            inputs = np.concatenate((inputs, data_mixed[0]), axis=1)
            output = output + shift
            feature_names += [f"mixed_feature_{i}" for i in range(data_mixed[0].shape[1])]


        x_train, x_test, y_train, y_test, _, noise_std_test = train_test_split(
            inputs, output, data_noise[1]+noise_scaler*data_mixed[1], test_size=n_test, random_state=1
        )  # We need the noise_std_test to compare with the pnn uncertainties
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=1
        )

        # normalize target

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train)
        y_val = scaler.transform(y_val)
        y_test = scaler.transform(y_test)
        y_train = y_train.flatten()
        y_val = y_val.flatten()
        y_test = y_test.flatten()

        y_means = scaler.mean_
        y_stds = scaler.scale_
        np.savez(
            f"{save_dir}data_{identifier}.npz",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_val=x_val,
            y_val=y_val,
            y_means=y_means,
            y_stds=y_stds,
            noise_std_test=noise_std_test,
            feature_names=feature_names,
        )
    else:
        npz = np.load(f"{save_dir}data_{identifier}.npz")
        x_train = npz["x_train"]
        y_train = npz["y_train"]
        x_val = npz["x_val"]
        y_val = npz["y_val"]
        x_test = npz["x_test"]
        y_test = npz["y_test"]
        y_means = npz["y_means"]
        y_stds = npz["y_stds"]
        noise_std_test = npz["noise_std_test"]
        feature_names = npz["feature_names"]
    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_means,
        y_stds,
        feature_names,
        noise_std_test,
    )


def train_pnn(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    identifier,
    save_dir,
    overwrite,
    beta_gaussian=False,
    small_model=False,
):
    input_dim = x_train.shape[1]

    cuda = torch.cuda.is_available()
    model_kwargs = {
        "n_features": input_dim,
        "n_units_per_layer": [64, 32] if small_model else [64, 64, 64, 32],
        "dropout_prob": 0.1,
        "cuda": cuda,
        "beta_gaussian": beta_gaussian,
    }

    model = ProbabilisticFeedForwardNetwork(**model_kwargs)
    trainer_params = {
        "progress_bar_refresh_rate": 0,
        "max_epochs": 200,
        "enable_model_summary": False,
    }
    model.mse_only = True
    model.fit(
        X_train=x_train if isinstance(x_train, np.ndarray) else x_train.cpu().detach().numpy(),
        y_train=y_train if isinstance(y_train, np.ndarray) else y_train.cpu().detach().numpy(),
        X_eval=x_val if isinstance(x_val, np.ndarray) else x_val.cpu().detach().numpy(),
        y_eval=y_val if isinstance(y_val, np.ndarray) else y_val.cpu().detach().numpy(),
        batch_size=64,
        patience=20,
        adversarial_training=False,
        trainer_params=trainer_params,
    )

    model.eval()
    x_test = x_test if isinstance(x_test, np.ndarray) else x_test.cpu().detach().numpy()
    y_test = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().detach().numpy()
    mse_before = np.mean(np.square(model.predict_target(x_test) - y_test))
    model.train()

    # finetune the model with negative gaussian log likelihood
    model.mse_only = False
    trainer_params["max_epochs"] = 100
    model.fit(
        X_train=x_train if isinstance(x_train, np.ndarray) else x_train.cpu().detach().numpy(),
        y_train=y_train if isinstance(y_train, np.ndarray) else y_train.cpu().detach().numpy(),
        X_eval=x_val if isinstance(x_val, np.ndarray) else x_val.cpu().detach().numpy(),
        y_eval=y_val if isinstance(y_val, np.ndarray) else y_val.cpu().detach().numpy(),
        batch_size=64,
        patience=40,
        adversarial_training=False,
        adversarial_epsilon=0.0005,
        trainer_params=trainer_params,
    )
    model = ProbabilisticFeedForwardNetwork.load_from_checkpoint(
        model.checkpoint_callback.best_model_path, **model_kwargs
    )
    model.eval()
    x_test = x_test if isinstance(x_test, np.ndarray) else x_test.cpu().detach().numpy()
    y_test = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().detach().numpy()
    mse_after = np.mean(np.square(model.predict_target(x_test) - y_test))

    # save the model
    if not os.path.exists(f"{save_dir}/models/pnn_{identifier}"):
        os.makedirs(f"{save_dir}/models/pnn_{identifier}")
    if overwrite:
        print("Saving checkpoint")
        torch.save(
            model.state_dict(), f"{save_dir}/models/pnn_{identifier}/pnn_state_dict.pth"
        )
    # np.save(
    #     f"{save_dir}models/pnn_{identifier}/pnn_test_mses.npy",
    #     np.array([mse_before, mse_after]),
    # )
    return model


def clue_explain(
    pnn,
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    ind_instances_to_explain,
    identifier,
    var_names,
    save_dir=None,
    save=False,
    sort=True,
):
    # code partly adapted from https://github.com/cambridge-mlg/CLUE
    trainset = Datafeed(x_train, y_train, transform=None)
    valset = Datafeed(x_val, y_val, transform=None)
    testset = Datafeed(x_test, torch.zeros(len(x_test)), transform=None)  # dummy labels
    if not os.path.exists(f"{save_dir}models/"):
        os.makedirs(f"{save_dir}models/")
    save_dir_vae = f"{save_dir}models/VAE_{identifier}"
    if not os.path.exists(save_dir_vae):
        os.makedirs(save_dir_vae)

    width = 300
    depth = 3  # number of hidden layers
    latent_dim = 6

    batch_size = 128

    nb_epochs = 2500
    lr = 1.5e-4
    early_stop = 10

    cuda = torch.cuda.is_available()
    if cuda:
        pnn.to("cuda")

    VAE = VAE_gauss_net(
        x_train.shape[1], width, depth, latent_dim, pred_sig=False, lr=lr, cuda=cuda
    )

    # clear out old models
    path = f"{save_dir_vae}_models"
    try:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except FileNotFoundError as e:
        pass

    vlb_train, vlb_dev = train_VAE(
        VAE,
        save_dir_vae,
        batch_size,
        nb_epochs,
        trainset,
        valset,
        cuda=cuda,
        flat_ims=False,
        train_plot=False,
        early_stop=early_stop,
    )

    VAE.load(f"{save_dir_vae}_models/theta_best.dat")

    # map to latent space and get ucnertainties
    (
        tr_aleatoric_vec,
        tr_epistemic_vec,
        z_train,
        x_train,
        y_train,
    ) = latent_project_gauss(
        pnn, VAE, dset=trainset, batch_size=2048, cuda=cuda, prob_BNN=False
    )

    te_aleatoric_vec, te_epistemic_vec, z_test, x_test, y_test = latent_project_gauss(
        pnn, VAE, dset=testset, batch_size=2048, cuda=cuda, prob_BNN=False
    )

    torch.cuda.empty_cache()

    z_init_batch = z_test[ind_instances_to_explain]
    x_init_batch = x_test[ind_instances_to_explain]

    torch.cuda.empty_cache()

    dist = Ln_distance(n=1, dim=(1))
    x_dim = x_init_batch.reshape(x_init_batch.shape[0], -1).shape[1]

    # we explain aleatory uncertainty in this work
    aleatoric_weight = 1
    epistemic_weight = 0
    uncertainty_weight = 0

    distance_weight = 1.5 / x_dim
    prediction_similarity_weight = 0

    mu_vec, std_vec = pnn.predict(x_init_batch, grad=False)
    desired_preds = mu_vec.cpu().numpy()
    if cuda:
        VAE.model.to("cuda")
    CLUE_explainer = CLUE(
        VAE,
        pnn,
        x_init_batch,
        uncertainty_weight=uncertainty_weight,
        aleatoric_weight=aleatoric_weight,
        epistemic_weight=epistemic_weight,
        prior_weight=0,
        distance_weight=distance_weight,
        latent_L2_weight=0,
        prediction_similarity_weight=prediction_similarity_weight,
        lr=1e-2,
        desired_preds=None,
        cond_mask=None,
        distance_metric=dist,
        z_init=z_init_batch,
        norm_MNIST=False,
        flatten_BNN=False,
        regression=True,
        cuda=False,
        prob_BNN=False,
    )

    (
        z_vec,
        x_vec,
        uncertainty_vec,
        epistemic_vec,
        aleatoric_vec,
        cost_vec,
        dist_vec,
    ) = CLUE_explainer.optimise(min_steps=3, max_steps=35, n_early_stop=3)

    importances = []
    differences = []
    for Nsample in range(len(ind_instances_to_explain)):
        difference = x_vec[-1, Nsample, :] - x_init_batch[Nsample]
        importance = np.abs(difference)
        importances.append(importance)
        differences.append(difference)
    differences = np.array(differences)
    differences = np.stack(differences, axis=0)

    importances = np.array(importances)
    importances = np.stack(importances, axis=0)
    pnn.to("cpu")
    x_test = x_test[ind_instances_to_explain]
    index_sorted = np.argsort(pnn.predict_uncertainty(x_test), kind="stable")
    importances_sorted = importances[index_sorted]
    differences_sorted = differences[index_sorted]
    instances_to_explain = x_test[index_sorted]
    if save:
        if not (os.path.exists(f"{save_dir}/importances")):
            os.makedirs(f"{save_dir}/importances")
        np.save(
            f"{save_dir}/importances/CLUE_importances_{identifier}.npy",
            {
                "importances_directed": differences,
                "feature_importance": importances_sorted,
                "var_names": var_names,
                "instances_to_explain": instances_to_explain,
            },
        )
    else:
        return {
            "importances_directed": differences_sorted if sort else differences,
            "feature_importance": importances_sorted if sort else importances,
            "instances_to_explain": instances_to_explain if sort else x_test,
        }


def varx_explain(
    pnn,
    x_train,
    instances_to_explain,
    identifier,
    var_names,
    save_dir=None,
    save=False,
    sort=True,
):
    n_background = 200
    n_reeval = 250
    random_instances = x_train[
        np.random.choice(x_train.shape[0], n_background, replace=False)
    ]
    # TODO implement LRP with captum
    pnn.to("cpu")
    explainer = shap.KernelExplainer(
        model=pnn.predict_uncertainty,
        data=random_instances,
        link="identity",
        silent=True,
    )

    x_test_sorted = instances_to_explain[
        np.argsort(pnn.predict_uncertainty(instances_to_explain))
    ]

    shap_values = explainer.shap_values(
        X=x_test_sorted if sort else instances_to_explain, nsamples=n_reeval
    )

    feature_importances = np.abs(shap_values)
    if save:
        if not (os.path.exists(f"{save_dir}/importances")):
            os.makedirs(f"{save_dir}/importances")

        np.save(
            f"{save_dir}/importances/VarX_importances_{identifier}.npy",
            {
                "importance_directed": shap_values,
                "feature_importance": feature_importances,
                "var_names": var_names,
                "instances_to_explain": x_test_sorted,
            },
        )
    else:
        return {
            "importance_directed": shap_values,
            "feature_importance": feature_importances,
            "instances_to_explain": x_test_sorted if sort else instances_to_explain,
        }


def varx_lrp_explain(
    pnn,
    instances_to_explain,
    identifier,
    var_names,
    save_dir=None,
    save=False,
    sort=True,
):
    from captum.attr import LRP

    lrp = LRP(pnn.model)

    instances_to_explain = torch.tensor(instances_to_explain, dtype=torch.float32).to(
        "cuda"
    )

    instances_to_explain_sorted = instances_to_explain[
        torch.argsort(pnn.predict_uncertainty_tensor(instances_to_explain), stable=True)
    ]

    # print(x_test.shape)
    attribution = (
        lrp.attribute(
            instances_to_explain_sorted.to("cuda")
            if sort
            else instances_to_explain.to("cuda"),
            target=1,
        )
        .cpu()
        .detach()
        .numpy()
    )
    feature_importances = np.abs(attribution)
    if save:
        if not (os.path.exists(f"{save_dir}/importances")):
            os.makedirs(f"{save_dir}/importances")

        np.save(
            f"{save_dir}/importances/VarXLRP_importances_{identifier}.npy",
            {
                "feature_importance": feature_importances,
                "importance_directed": attribution,
                "var_names": var_names,
                "instances_to_explain": instances_to_explain_sorted.cpu().detach().numpy()
                if sort
                else instances_to_explain.cpu().detach().numpy(),
            },
        )
    else:
        return {
            "feature_importance": feature_importances,
            "importance_directed": attribution,
            "var_names": var_names,
            "instances_to_explain": instances_to_explain_sorted.cpu().detach().numpy()
            if sort
            else instances_to_explain.cpu().detach().numpy(),
        }


def varx_ig_explain(
    pnn,
    instances_to_explain,
    identifier,
    var_names,
    save_dir=None,
    save=False,
    sort=True,
):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(pnn.model)

    instances_to_explain = torch.tensor(instances_to_explain, dtype=torch.float32).to(
        "cuda"
    )

    instances_to_explain_sorted = instances_to_explain[
        torch.argsort(pnn.predict_uncertainty_tensor(instances_to_explain), stable=True)
    ]

    attribution, _ = ig.attribute(
        instances_to_explain_sorted.to("cuda")
        if sort
        else instances_to_explain.to("cuda"),
        target=1,
        method="gausslegendre",
        return_convergence_delta=True,
    )
    attribution = attribution.cpu().detach().numpy()
    feature_importances = np.abs(attribution)

    if save:
        if not (os.path.exists(f"{save_dir}/importances")):
            os.makedirs(f"{save_dir}/importances")

        np.save(
            f"{save_dir}/importances/VarXIG_importances_{identifier}.npy",
            {
                "feature_importance": feature_importances,
                "importance_directed": attribution,
                "var_names": var_names,
                "instances_to_explain": instances_to_explain_sorted.cpu().detach().numpy()
                if sort
                else instances_to_explain.cpu().detach().numpy(),
            },
        )
    else:
        return {
            "feature_importance": feature_importances,
            "importance_directed": attribution,
            "var_names": var_names,
            "instances_to_explain": instances_to_explain_sorted.cpu().numpy()
            if sort
            else instances_to_explain.cpu().numpy(),
        }


def explain_mean(pnn, x_train, instances_to_explain, identifier, save_dir, var_names):
    n_background = 200
    n_reeval = 250

    random_instances = x_train[
        np.random.choice(x_train.shape[0], n_background, replace=False)
    ]
    pnn.to("cpu")
    explainer = shap.KernelExplainer(
        model=pnn.predict_target,
        data=random_instances,
        link="identity",
        silent=True,
    )
    if torch.cuda.is_available():
        pnn.to("cuda")
    shap_values = explainer.shap_values(X=instances_to_explain, nsamples=n_reeval)

    feature_importances = np.abs(shap_values)

    if not (os.path.exists(f"{save_dir}/importances")):
        os.makedirs(f"{save_dir}/importances")

    np.save(
        f"{save_dir}/importances/mean_importances_{identifier}.npy",
        {
            "importances_directed": shap_values,
            "feature_importance": feature_importances,
            "var_names": var_names,
            "instances_to_explain": instances_to_explain,
        },
    )


def load_importances(
    method: str, identifier: str, save_dir: str, uncertainty_ind: str = None, i: int = 0
):
    uncertainty_ind = f"_{uncertainty_ind}"

    return np.load(
        f"{save_dir}importances/{method}_importances_{identifier}_run_{i}{uncertainty_ind}.npy",
        allow_pickle=True,
    ).item()


def save_importance_data(
    importance,
    identifier,
    method=None,
    uncertainty_ind: str = None,
    save_dir=None,
):
    feature_importances = importance["feature_importance"]
    feature_importances = feature_importances.mean(axis=0)
    var_names = importance["var_names"]
    feature_importances = pd.DataFrame(
        {"feature_importance": feature_importances}, index=var_names
    )
    feature_importances = feature_importances.sort_values(
        by="feature_importance", ascending=False
    )
    feature_importances.to_csv(
        f"{save_dir}/{method}_{uncertainty_ind}_importances_{identifier}.csv",
        index_label="feature_name",
    )


def run_uncertainty_explanation_experiment(
    n_instances_to_explain=200,
    explainer_repeats=1,
    noise_scaler=2.0,
    n=40000,
    n_test=1500,
    k_mixed=0,
    remake_data=True,
    overwrite_pnn=True,
    beta_gaussian=False,
):
    save_dir = f"results/new_synthetic/"
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    identifier = (
        f"n_{n}_s_{noise_scaler:.2f}_n_test_{n_test}_n_exp_{n_instances_to_explain}"
    )

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # create or load a heteroscedastic dataset with non linear noise model and linear mean model
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        y_means,
        y_stds,
        var_names,
        noise_std_test,
    ) = get_data(
        remake_data=remake_data,
        n_train=n,
        n_test=n_test,
        k_mixed=k_mixed,
        noise_scaler=noise_scaler,
        save_dir=save_dir,
        identifier=identifier,
    )
    var_names = list(var_names)
    dtype = torch.float32
    x_train = torch.tensor(x_train, dtype=dtype)
    y_train = torch.tensor(y_train, dtype=dtype)
    x_val = torch.tensor(x_val, dtype=dtype)
    y_val = torch.tensor(y_val, dtype=dtype)
    x_test = torch.tensor(x_test, dtype=dtype)
    y_test = torch.tensor(y_test, dtype=dtype)
    if beta_gaussian:
        identifier += "_bgll"
    # train the heteroscedastic gaussian neural network
    pnn = train_pnn(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        identifier=identifier,
        save_dir=save_dir,
        overwrite=overwrite_pnn,
        beta_gaussian=beta_gaussian,
    )

    # predict and save the test set output of the pnn
    mu_test, sigma_test = pnn.predict(x_test)
    mu_test = mu_test.cpu().detach().numpy()
    sigma_test = sigma_test.cpu().detach().numpy()
    out = {
        "mu_test": mu_test,
        "sigma_test": sigma_test,
        "y_test": y_test,
        "noise_std_test": noise_std_test,
    }
    pd.DataFrame(out).to_csv(
        f"{save_dir}/models/pnn_{identifier}/pnn_test_output.csv", index=False
    )

    if not (os.path.exists(f"{save_dir}/models/xgb_{identifier}")):
        os.makedirs(f"{save_dir}/models/xgb_{identifier}")
    xgboost_model = train_xgboost(x_train, y_train)
    prediction_test = xgboost_model.predict(x_test)
    prediction_val = xgboost_model.predict(x_val)

    error = y_val - prediction_val
    xgboost_error_model = train_xgboost_var(error=error, x_val=x_val)
    uncertainty = xgboost_error_model.predict(x_test)
    pd.DataFrame(
        {
            "prediction_test": prediction_test,
            "y_test": y_test,
            "uncertainty_test": uncertainty,
        }
    ).to_csv(f"{save_dir}/models/xgb_{identifier}/xgb_test_output.csv", index=False)

    # get the indices of the instances to explain
    # (We want to explain high, low uncertainty instances and random instances)
    high_uncertainty_ind = np.argsort(sigma_test)[-n_instances_to_explain:]
    low_uncertainty_ind = np.argsort(sigma_test)[:n_instances_to_explain]
    random_ind = np.random.choice(
        x_test.shape[0], n_instances_to_explain, replace=False
    )

    high_uncertainty_ind_is = np.argsort(uncertainty)[-n_instances_to_explain:]
    low_uncertainty_ind_is = np.argsort(uncertainty)[:n_instances_to_explain]
    indices_is = {
        "highU": high_uncertainty_ind_is,
        "lowU": low_uncertainty_ind_is,
        "randomU": random_ind,
    }

    # measure time with timeit
    varx_elapsed = []
    clue_elapsed = []
    varxlrp_elapsed = []
    varxig_elapsed = []
    infoshap_elapsed = []

    for indices, ind_identifier in zip(
        [high_uncertainty_ind, low_uncertainty_ind, random_ind],
        ["highU", "lowU", "randomU"],
    ):
        print(f"Running {ind_identifier}")
        for i in range(explainer_repeats):
            print(f"Running repeat {i}")
            identifier_run = f"{identifier}_run_{i}_{ind_identifier}"
            explain_mean(
                pnn=pnn,
                x_train=x_train.cpu().detach().numpy(),
                instances_to_explain=x_test.cpu().detach().numpy()[indices],
                identifier=identifier_run,
                save_dir=save_dir,
                var_names=var_names,
            )
            start = timer()
            
            varx_explain(
                pnn=pnn,
                x_train=x_train.cpu().detach().numpy(),
                instances_to_explain=x_test.cpu().detach().numpy()[indices],
                identifier=identifier_run,
                save_dir=save_dir,
                var_names=var_names,
                save=True,
                sort=False,
            )
            end = timer()
            varx_elapsed.append(end - start)

            start = timer()
            clue_explain(
                pnn=pnn,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                ind_instances_to_explain=indices,
                identifier=identifier_run,
                save_dir=save_dir,
                var_names=var_names,
                save=True,
                sort=False,
            )
            end = timer()
            clue_elapsed.append(end - start)

            start = timer()
            varx_ig_explain(
                pnn=pnn,
                instances_to_explain=x_test.cpu().detach().numpy()[indices],
                identifier=identifier_run,
                save_dir=save_dir,
                var_names=var_names,
                save=True,
                sort=False,
            )
            end = timer()
            varxig_elapsed.append(end - start)

            start = timer()
            varx_lrp_explain(
                pnn=pnn,
                instances_to_explain=x_test.cpu().detach().numpy()[indices],
                identifier=identifier_run,
                save_dir=save_dir,
                save=True,
                sort=False,
                var_names=var_names,
            )
            end = timer()
            varxlrp_elapsed.append(end - start)

            start = timer()

            infoboost_explain(
                model=xgboost_error_model,
                x_val=x_val.cpu().detach().numpy(),
                y_val=y_val.cpu().detach().numpy(),
                instances_to_explain=x_test.cpu().detach().numpy()[indices_is[ind_identifier]],
                identifier=identifier_run,
                save_dir=save_dir,
                var_names=var_names + ["bias"],
                save=True,
                sort=False,
            )
            end = timer()
            infoshap_elapsed.append(end - start)

            

    for method in ["VarX", "CLUE", "VarXIG", "VarXLRP", "infoshap"]:
        for u in ["highU", "lowU", "randomU"]:
            importance = load_importances(
                method=method,
                identifier=identifier,
                save_dir=save_dir,
                uncertainty_ind=u,
                i=0,
            )
            save_importance_data(
                importance,
                identifier=identifier,
                method=method,
                uncertainty_ind=u,
                save_dir="../plotting/data/importances",
            )

    if not (os.path.exists(save_dir + "/comptime/")):
        os.makedirs(save_dir + "/comptime/")
    pd.DataFrame(
        {
            "varx_elapsed": varx_elapsed,
            "clue_elapsed": clue_elapsed,
            "varxlrp_elapsed": varxlrp_elapsed,
            "varxig_elapsed": varxig_elapsed,
            "infoshap_elapsed": infoshap_elapsed,
        }
    ).to_csv(save_dir + f"/comptime/{identifier}_time_elapsed.csv")


def get_explanation(
    model,
    explain_method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sort: bool = True,
    metric=None,
    return_xgboost_error_model: bool = False,
    only_explanation: bool = False,
) -> float:
    """
    :param model: The model to explain
    :param explain_method: The explanation method to use "varx", "clue", "infoshap"
    :param X: A numpy array of shape (N, D)
    :param y: A numpy array of shape (N, 1)
    :x_val: A numpy array of shape (N, D) or None, only used for CLUE
    :y_val: A numpy array of shape (N, 1) or None , only used for CLUE
    :return: explanation
    """
    assert not (only_explanation and return_xgboost_error_model)
    if explain_method == "varx":
        kwargs = {
            "pnn": model,
            "x_train": x_train,
            "instances_to_explain": x_test,
            "identifier": None,
            "save_dir": None,
            "save": False,
            "var_names": list(range(x_train.shape[1])),
            "sort": sort,
        }
        temp = varx_explain(**kwargs)
        explanation = temp["importance_directed"]
        x_test_sorted = temp["instances_to_explain"]
    elif explain_method == "varx_lrp":
        kwargs = {
            "pnn": model,
            "instances_to_explain": x_test,
            "identifier": None,
            "save_dir": None,
            "save": False,
            "var_names": list(range(x_train.shape[1])),
            "sort": sort,
        }
        temp = varx_lrp_explain(**kwargs)
        explanation = temp["importance_directed"]
        x_test_sorted = temp["instances_to_explain"]
    elif explain_method == "varx_ig":
        kwargs = {
            "pnn": model,
            "instances_to_explain": x_test,
            "identifier": None,
            "save_dir": None,
            "save": False,
            "var_names": list(range(x_train.shape[1])),
            "sort": sort,
        }
        temp = varx_ig_explain(**kwargs)
        explanation = temp["importance_directed"]
        x_test_sorted = temp["instances_to_explain"]
    elif explain_method == "clue":
        assert x_val is not None and y_val is not None
        kwargs = {
            "pnn": model,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "ind_instances_to_explain": list(range(len(x_test))),
            "identifier": metric,
            "save_dir": metric,
            "save": False,
            "var_names": list(range(x_train.shape[1])),
            "sort": sort,
        }
        print()
        temp = clue_explain(**kwargs)
        explanation = temp["importances_directed"]
        x_test_sorted = temp["instances_to_explain"]

    elif explain_method == "infoshap":
        kwargs = {
            "model": model,
            "x_val": x_train,
            "y_val": y_train,
            "instances_to_explain": x_test,
            "var_names": list(range(x_train.shape[1])) + ["bias"],
            "sort": sort,
            "return_model": True,
            "save": False,
        }
        temp = infoboost_explain(**kwargs)
        explanation = temp["importance_directed"]
        x_test_sorted = temp["instances_to_explain"]
        if return_xgboost_error_model:
            return explanation, x_test_sorted, temp["error_model"]

    else:
        raise ValueError("Unknown explanation method")

    if only_explanation:
        return explanation
    else:
        return explanation, x_test_sorted
