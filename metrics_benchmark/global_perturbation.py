from typing import List, Optional
import os
import inspect
import sys
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.experiment_utils import train_pnn, get_explanation
from utils.infoshap_xgboost import train_xgboost


def perturb_top_k_global_features(
    x: np.ndarray,
    explanation: np.ndarray,
    k=1,
    epsilon=0.01,
) -> np.ndarray:
    """
    :param x: The instance to perturb
    :param explanation: The explanation of the instance
    :param k: The number of features to perturb
    :param direction: The direction to perturb the features "positive" or "negative"
                        (positive increases the feature value which contributes positvely to the prediction value,
                        negative increases the feature value which contributes negative to the prediction value)
    :param epsilon: The perturbation size
    :return: The perturbed instance
    """
    x_new = x.copy()
    explanation = np.abs(explanation)
    assert explanation.shape == x.shape
    top_k_features = np.argsort(np.mean(explanation, axis=0))[
        -k:
    ]  # top k uncertainty sources
    noise = np.random.normal(0, epsilon, size=(x_new.shape[0], k))
    x_new[:, top_k_features] += noise

    return x_new


def get_global_perturbation_delta(
    model,
    explain_method,
    x_train,
    x_test,
    y_test,
    y_train,
    x_val,
    y_val,
    epsilon,
    top_k,
):
    """
    :param model: The model to explain
    :param explain_method: The explanation method to use "varx", "clue", "infoshap"
    :param X: A numpy array of shape (N, D)
    :param y: A numpy array of shape (N, 1)
    :return: explanation
    """
    if explain_method != "infoshap":
        model.to("cuda")
        explanation, x_test = get_explanation(
            model=model,
            explain_method=explain_method,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            x_val=x_val,
            metric="perturbation",
            sort=False,
        )
    else:
        explanation, x_test, error_model = get_explanation(
            model=model,
            explain_method=explain_method,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            x_val=x_val,
            metric="perturbation",
            sort=False,
            return_xgboost_error_model=True,
        )

    x_perturbed = perturb_top_k_global_features(
        x_test.copy(),
        explanation,
        k=top_k,
        epsilon=epsilon,
    )

    if explain_method == "infoshap":
        target = model.predict(x_test)
        uncertainty = error_model.predict(x_test)
        uncertainty_perturbed = error_model.predict(x_perturbed)

    else:
        model.to("cpu")
        target = model.predict_target(x_test)
        uncertainty = model.predict_uncertainty(x_test)
        uncertainty_perturbed = model.predict_uncertainty(x_perturbed)

    residuals = np.square(target - y_test)
    # mse of top 10% most certain instances
    residuals_top = residuals[np.argsort(uncertainty)][: int(len(residuals) * 0.1)]
    residuals_top_perturbed = residuals[np.argsort(uncertainty_perturbed)][
        : int(len(residuals) * 0.1)
    ]

    from sklearn.metrics import dcg_score, ndcg_score
    from scipy.stats import pearsonr, spearmanr

    dcg_before = dcg_score([residuals], [uncertainty])
    dcg_perturbed = dcg_score([residuals], [uncertainty_perturbed])
    ndcg_before = ndcg_score([residuals], [uncertainty])
    ndcg_perturbed = ndcg_score([residuals], [uncertainty_perturbed])
    correlation = pearsonr(uncertainty, residuals)[0]
    spearman = spearmanr(uncertainty, residuals)[0]
    correlation_perturbed = pearsonr(uncertainty_perturbed, residuals)[0]
    spearman_perturbed = spearmanr(uncertainty_perturbed, residuals)[0]
    return (
        (np.mean(residuals_top_perturbed) - np.mean(residuals_top))
        / np.mean(residuals_top),
        dcg_before,
        dcg_perturbed,
        ndcg_before,
        ndcg_perturbed,
        correlation,
        correlation_perturbed,
        spearman,
        spearman_perturbed,
    )


def perturbation_experiment(
    epsilons: list,
    top_k,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    explain_methods: Optional[List] = None,
    small_model: bool = False,
) -> list:
    """
    :param epsilons: The perturbation sizes
    :param direction: The direction to perturb the features "positive" or "negative"
                        (positive increases the feature value which contributes positvely to the prediction value,
                        negative increases the feature value which contributes negative to the prediction value)
    :param top_k: The number of features to perturb
    :param explain_methods: The explanation methods to use "varx", "clue", "infoshap", varx_lrp"
    :param x_train: The training data
    :param x_val: The validation data
    :param x_test: The test data
    :param y_train: The training labels
    :param y_val: The validation labels
    :param y_test: The test labels
    :return: pertubation differences
    """
    if explain_methods is None:
        explain_methods = ["varx_ig", "varx_lrp", "varx", "clue", "infoshap"]
    perturbation_deltas = {}
    pnn_model = train_pnn(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        identifier="perturbation",
        save_dir="perturbation",
        overwrite=True,
        beta_gaussian=False,
        small_model=small_model,
    )
    xgboost_model = train_xgboost(x_train, y_train)

    for model_ident in explain_methods:
        perturbation_deltas[model_ident] = {}
        for epsilon in epsilons:
            perturbation_deltas[model_ident][epsilon] = {}
            if model_ident in {"varx_ig", "varx_lrp", "varx", "clue"}:
                model = pnn_model
            else:
                model = xgboost_model

            (
                residual_change,
                dcg_before,
                dcg_perturbed,
                ndcg_before,
                ndcg_perturbed,
                correlation_before,
                correlation_perturbed,
                spearman_before,
                spearman_perturbed,
            ) = get_global_perturbation_delta(
                model=model,
                explain_method=model_ident,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                y_val=y_val,
                x_val=x_val,
                epsilon=epsilon,
                top_k=top_k,
            )

            perturbation_deltas[model_ident][epsilon][
                "residual_change"
            ] = residual_change
            perturbation_deltas[model_ident][epsilon]["dcg_before"] = dcg_before
            perturbation_deltas[model_ident][epsilon]["dcg_perturbed"] = dcg_perturbed
            perturbation_deltas[model_ident][epsilon]["ndcg_before"] = ndcg_before
            perturbation_deltas[model_ident][epsilon]["ndcg_perturbed"] = ndcg_perturbed
            perturbation_deltas[model_ident][epsilon][
                "correlation_before"
            ] = correlation_before
            perturbation_deltas[model_ident][epsilon][
                "correlation_perturbed"
            ] = correlation_perturbed
            perturbation_deltas[model_ident][epsilon][
                "spearman_before"
            ] = spearman_before
            perturbation_deltas[model_ident][epsilon][
                "spearman_perturbed"
            ] = spearman_perturbed

    return perturbation_deltas
