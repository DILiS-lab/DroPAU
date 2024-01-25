import os
import inspect
import sys
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.experiment_utils import train_pnn, get_explanation
from utils.infoshap_xgboost import train_xgboost
from lipschitz_metric import get_Ls_for_dataset


def get_lipschitz_robustness(
    model,
    explain_method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    dataset: str,
) -> float:
    """
    :param model: The model to explain
    :param explain_method: The explanation method to use "varx", "clue", "infoshap"
    :param X: A numpy array of shape (N, D)
    :param y: A numpy array of shape (N, 1)
    :param epsilon: The perturbation size
    :return: The robustness of the model:
    absolute difference between the prediction on the original data and the prediction on the perturbed data
    """

    return get_Ls_for_dataset(
        x_test=x_test,
        x_train=x_train,
        explainer_func=get_explanation,
        num_perturbations=100,
        number_of_samples=200,
        approx_perturbations=True if dataset == "lsat" else False,
        model=model,
        explain_method=explain_method,
        y_train=y_train,
        y_val=y_val,
        x_val=x_val,
        sort=False,
        only_explanation=True,
        metric=f"robustness_lipschitz_{dataset}_{explain_method}",
    )


def robustness_experiment(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    dataset: str,
) -> list:
    """
    :param epsilons: The perturbation sizes
    :param x_train: A numpy array of shape (N, D)
    :param x_test: A numpy array of shape (N, D)
    :param x_val: A numpy array of shape (N, D)
    :param y_train: A numpy array of shape (N, 1)
    :param y_val: A numpy array of shape (N, 1)
    :param y_test: A numpy array of shape (N, 1)
    :return: A list of robustness values for each epsilon
    """
    var_xai_methods = ["varx_ig", "varx_lrp", "varx", "clue", "infoshap"]
    robustness_values = {}
    # Models are trained outside to loop to use same model for all xai_methods (where applicable)
    pnn_model = train_pnn(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        identifier=f"robustness_lipschitz_{dataset}",
        save_dir=f"robustness_lipschitz_{dataset}",
        overwrite=True,
        beta_gaussian=False,
    )
    xgboost_model = train_xgboost(x_train, y_train)

    for var_xai_method in var_xai_methods:
        robustness_values[var_xai_method] = {}
        if var_xai_method in {"varx_ig", "varx_lrp", "varx", "clue"}:
            model = pnn_model
        else:
            model = xgboost_model

        (
            robustness_values[var_xai_method]["L_out"],
            robustness_values[var_xai_method]["x0_out"],
            robustness_values[var_xai_method]["exp0_out"],
            robustness_values[var_xai_method]["x_worst_case_out"],
            robustness_values[var_xai_method]["exp_worst_case_out"],
        ) = get_lipschitz_robustness(
            model=model,
            explain_method=var_xai_method,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            dataset=dataset,
        )

    return robustness_values
