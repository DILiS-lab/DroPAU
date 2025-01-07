import os
import inspect
import sys
import numpy as np
from typing import List, Optional
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.experiment_utils import train_pnn, get_explanation
from utils.infoshap_xgboost import train_xgboost

def perturb_top_k_features(
    x: np.ndarray,
    explanation: np.ndarray,
    k=1,
    direction="positive",
    epsilon=0.01,
    warn=False,
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
    assert explanation.shape == x.shape
    new_instances = []
    for instance, instance_explanation in zip(x, explanation):
        if direction == "positive":
            # most positive feature explanation
            top_k_features = np.argsort(instance_explanation)[-k:]
        elif direction == "negative":
            # most negative feature explanation
            top_k_features = np.argsort(instance_explanation)[:k]
            # assert (top_k_features < 0).all()
        else:
            raise ValueError("Unknown direction")
        if warn and 3 in top_k_features:
            print("WARNING: FEATURE 3 IS BEING PERTURBED")
        instance[top_k_features] += epsilon
        # TODO consider relative perturbation
        new_instances.append(instance)
    x_new = np.stack(new_instances)
    return x_new


def get_perturbation_delta(
    model,
    explain_method,
    x_train,
    x_test,
    y_train,
    x_val,
    y_val,
    epsilon,
    direction,
    top_k,
    warn=False,
):
    """
    :param model: The model to explain
    :param explain_method: The explanation method to use "varx", "clue", "infoshap"
    :param X: A numpy array of shape (N, D)
    :param y: A numpy array of shape (N, 1)
    :return: The perturbation delta
    """
    if explain_method != "infoshap":
        model.to("cuda")
    explanation, x_test_sorted = get_explanation(
        model=model,
        explain_method=explain_method,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        x_val=x_val,
        metric="perturbation",
    )

    x_perturbed = perturb_top_k_features(
        x_test_sorted.copy(),
        explanation,
        k=top_k,
        direction=direction,
        epsilon=epsilon,
        warn=warn,
    )

    if explain_method == "infoshap":
        uncertainty = model.predict(x_test_sorted)
        uncertainty_perturbed = model.predict(x_perturbed)
    else:
        model.to("cpu")
        uncertainty = model.predict_uncertainty(x_test_sorted)
        uncertainty_perturbed = model.predict_uncertainty(x_perturbed)
    return (uncertainty - uncertainty_perturbed).tolist()


def perturbation_experiment(
    epsilons: list,
    direction: str,
    top_k,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    explain_methods: Optional[List] = None,
    warn=False,
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
    )
    xgboost_model = train_xgboost(x_train, y_train)

    for model_ident in explain_methods:
        perturbation_deltas[model_ident] = {}
        for epsilon in epsilons:
            if model_ident in {"varx_ig", "varx_lrp", "varx", "clue"}:
                model = pnn_model
            else:
                model = xgboost_model

            perturbation_deltas[model_ident][epsilon] = get_perturbation_delta(
                model=model,
                explain_method=model_ident,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                x_val=x_val,
                epsilon=epsilon,
                direction=direction,
                top_k=top_k,
                warn=warn,
            )

    return perturbation_deltas
