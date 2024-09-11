import os
import inspect
import sys
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.experiment_utils import (
    train_pnn,
    get_explanation,
    noise_model,
    simple_noise_model,
)
from utils.infoshap_xgboost import train_xgboost
import copy


def add_noise_features(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    noise_level,
    n_noise_features,
    use_simple_noise_model,
    n_noise_features_mixed=0
):
    if y_train.dtype == np.int32 or y_train.dtype == np.int64:
        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)
        y_val = y_val.astype(np.float64)

    n_train_val = len(x_train) + len(x_val)
    if use_simple_noise_model:
        features, stds = simple_noise_model(
            n=n_train_val + len(x_test), k=n_noise_features, noise_level=noise_level
        )
    else:
        features, stds = noise_model(
            n=n_train_val + len(x_test),
            k=n_noise_features,
            noise_level=noise_level,
        )
    y_train += np.random.normal(0, stds[: len(x_train)])
    y_val += np.random.normal(0, stds[len(x_train) : n_train_val])
    y_test += np.random.normal(0, stds[n_train_val:])

    if n_noise_features_mixed > 0:
        assert not(use_simple_noise_model), "Cannot use mixed noise model with simple noise model"
        # add noise features which also correlate with the target
        data_mixed = noise_model(
            n=(n_train_val + len(x_test)), k=n_noise_features_mixed, noise_level=noise_level
        )
        # standardize the data
        location_change = (data_mixed[1]-np.mean(data_mixed[1]))/np.std(data_mixed[1])
        # scale to the same range as the original data for meaningful contribution
        shift = np.random.normal(loc=location_change*1.2, scale=data_mixed[1], size=(n_train_val + len(x_test)))
        features = np.concatenate((features, data_mixed[0]), axis=1)
        y_train = y_train + shift[:len(x_train)]
        y_val = y_val + shift[len(x_train):n_train_val]
        y_test = y_test + shift[n_train_val:] 
    print(x_train.shape)
    x_train = np.concatenate([x_train, features[: len(x_train)]], axis=1)
    x_val = np.concatenate([x_val, features[len(x_train) : n_train_val]], axis=1)
    x_test = np.concatenate([x_test, features[n_train_val:]], axis=1)
    print(x_train.shape)

    

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def get_mass(explanation, n_noise_features):
    explanation = np.abs(explanation)
    noise_feature_mass = np.sum(explanation[-n_noise_features:]) / np.sum(explanation)
    return noise_feature_mass


def get_precision(explanation, n_noise_features):
    explanation = np.abs(explanation)  # feature importance
    top_k_features = np.argsort(explanation)[
        -n_noise_features:
    ]  # these are the noise features, since we append them to the end of the feature vector
    instance_precision = (
        np.sum(top_k_features >= (len(explanation) - n_noise_features))
        / n_noise_features
    )
    return instance_precision


def global_localization_precision(explanation: np.ndarray, n_noise_features: int):
    """
    :param explanation: The explanation values for all features
    :param n_noise_features: The number of noise features
    :return: The localization precision
    """
    explanation = np.mean(np.abs(explanation), axis=0)  # global explanation

    global_localization_precision = get_precision(explanation, n_noise_features)

    return global_localization_precision


def global_mass_accuracy(explanation: np.ndarray, n_noise_features: int):
    """
    :param explanation: The explanation values for all features
    :param n_noise_features: The number of noise features
    :return: The mass accuracy
    """
    explanation = np.mean(np.abs(explanation), axis=0)  # global explanation

    noise_feature_mass = get_mass(explanation, n_noise_features)
    return noise_feature_mass


def get_localization_metrics(
    model,
    explain_method,
    x_train,
    x_test,
    y_train,
    y_val,
    x_val,
    n_noise_features,
    dataset=None,
    **kwargs,
):  # based on relevance rank accuracy in clevr xai
    explanation, x_test_2 = get_explanation(
        model=model,
        explain_method=explain_method,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        x_val=x_val,
        sort=False,
        metric=f"localization_{dataset}_{explain_method}",
    )

    explanation = np.abs(explanation)  # this is the feature importance
    instance_precisions = []
    mass_accuracies = []
    n_instances = explanation.shape[0]
    n_instances10percent = int(n_instances * 0.1)
    explanation_top = explanation[-n_instances10percent:]
    explanation_bottom = explanation[:n_instances10percent]

    global_localization_precision_all = global_localization_precision(
        explanation=explanation, n_noise_features=n_noise_features
    )
    global_mass_accuracy_all = global_mass_accuracy(
        explanation=explanation, n_noise_features=n_noise_features
    )
    global_localization_precision_top = global_localization_precision(
        explanation=explanation_top, n_noise_features=n_noise_features
    )
    global_mass_accuracy_top = global_mass_accuracy(
        explanation=explanation_top, n_noise_features=n_noise_features
    )
    global_localization_precision_bottom = global_localization_precision(
        explanation=explanation_bottom, n_noise_features=n_noise_features
    )
    global_mass_accuracy_bottom = global_mass_accuracy(
        explanation=explanation_bottom, n_noise_features=n_noise_features
    )

    for instance_explanation in explanation:
        # check if last n_noise_features features have highest absolute explanation value
        instance_precision = get_precision(instance_explanation, n_noise_features)
        instance_precisions.append(instance_precision)

        noise_feature_mass = get_mass(instance_explanation, n_noise_features)

        mass_accuracies.append(noise_feature_mass)

    return {
        "instance_precisions": instance_precisions,
        "mass_accuracies": mass_accuracies,
        "global_localization_precision": global_localization_precision_all,
        "global_mass_accuracy": global_mass_accuracy_all,
        "global_localization_precision_top": global_localization_precision_top,
        "global_mass_accuracy_top": global_mass_accuracy_top,
        "global_localization_precision_bottom": global_localization_precision_bottom,
        "global_mass_accuracy_bottom": global_mass_accuracy_bottom,
    }


def localization_experiment(
    noise_levels: list,
    n_noise_features_list: list,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    use_simple_noise_model: bool = False,
    dataset: str = None,
    n_noise_features_mixed: int = 0
):
    """
    :param noise_levels: Overall magnitude of the added noise
    :param n_noise_features_list: The number of noise-causing features to add to the real data, which have to be found by the explanation method
    :param x_train: The training data
    :param x_val: The validation data
    :param x_test: The test data
    :param y_train: The training targets
    :param y_val: The validation targets
    :param y_test: The test targets
    :return: localization precision differences
    """
    if use_simple_noise_model:
        assert n_noise_features_mixed == 0, "Cannot use mixed noise model with simple noise model"

    model_indentifiers = ["varx_ig", "varx_lrp", "varx", "clue", "infoshap"]
    localization_precisions = {}
    mass_accuracies = {}
    global_metrics = {}
    for noise_level in noise_levels:
        localization_precisions[noise_level] = {}
        mass_accuracies[noise_level] = {}
        global_metrics[noise_level] = {}
        for n_noise_features in n_noise_features_list:
            localization_precisions[noise_level][n_noise_features+n_noise_features_mixed] = {}
            mass_accuracies[noise_level][n_noise_features+n_noise_features_mixed] = {}
            global_metrics[noise_level][n_noise_features+n_noise_features_mixed] = {}

            # print("y_train", y_train)
            # print("y_test", y_test)
            noised_dataset = add_noise_features(
                x_train=copy.deepcopy(x_train),
                y_train=copy.deepcopy(y_train),
                x_val=copy.deepcopy(x_val),
                y_val=copy.deepcopy(y_val),
                x_test=copy.deepcopy(x_test),
                y_test=copy.deepcopy(y_test),
                noise_level=noise_level,
                n_noise_features=n_noise_features,
                use_simple_noise_model=use_simple_noise_model,
                n_noise_features_mixed=n_noise_features_mixed
            )

            pnn_model = train_pnn(
                **noised_dataset,
                identifier=f"localization_{dataset}_{noise_level}_{n_noise_features}_mixed_{n_noise_features_mixed}",
                save_dir=f"localization_{dataset}_{noise_level}_{n_noise_features}_mixed_{n_noise_features_mixed}",
                overwrite=True,
                beta_gaussian=False,
            )
            xgboost_model = train_xgboost(
                noised_dataset["x_train"], noised_dataset["y_train"]
            )

            for model_ident in model_indentifiers:
                if model_ident in {"varx_ig", "varx_lrp", "varx", "clue"}:
                    model = pnn_model
                else:
                    model = xgboost_model

                result = get_localization_metrics(
                    **noised_dataset,
                    model=model,
                    explain_method=model_ident,
                    n_noise_features=n_noise_features+n_noise_features_mixed,
                    dataset=f"{dataset}_{noise_level}_{n_noise_features}",
                )
                precisions = result["instance_precisions"]
                accuracies = result["mass_accuracies"]
                global_localization_precision = result["global_localization_precision"]
                global_mass_accuracy = result["global_mass_accuracy"]
                global_mass_accuracy_top = result["global_mass_accuracy_top"]
                global_localization_precision_top = result[
                    "global_localization_precision_top"
                ]
                global_mass_accuracy_bottom = result["global_mass_accuracy_bottom"]
                global_localization_precision_bottom = result[
                    "global_localization_precision_bottom"
                ]
                global_metrics[noise_level][n_noise_features+n_noise_features_mixed][model_ident] = {
                    "global_localization_precision": global_localization_precision,
                    "global_mass_accuracy": global_mass_accuracy,
                    "global_mass_accuracy_top": global_mass_accuracy_top,
                    "global_localization_precision_top": global_localization_precision_top,
                    "global_mass_accuracy_bottom": global_mass_accuracy_bottom,
                    "global_localization_precision_bottom": global_localization_precision_bottom,
                }
                localization_precisions[noise_level][n_noise_features+n_noise_features_mixed][
                    model_ident
                ] = precisions
                mass_accuracies[noise_level][n_noise_features+n_noise_features_mixed][model_ident] = accuracies

    # shutil.rmtree("lightning_logs")

    return localization_precisions, mass_accuracies, global_metrics
