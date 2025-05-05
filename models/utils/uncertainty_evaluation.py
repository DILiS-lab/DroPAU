import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from matplotlib import pyplot as plt
import pandas as pd


def uncertainty_quality_metrics(
    y_prediction: np.ndarray, y_target: np.ndarray, y_uncertainty: np.ndarray
) -> dict:
    """
    Calculates uncertainty quality metrics for the given prediction and target values.
    """
    squared_error = np.square(y_prediction - y_target)
    absolute_error = np.abs(y_prediction - y_target)

    pearson_sq = pearsonr(squared_error, y_uncertainty)[0]
    spearman_sq = spearmanr(squared_error, y_uncertainty)[0]
    pearson_abs = pearsonr(absolute_error, y_uncertainty)[0]
    spearman_abs = spearmanr(absolute_error, y_uncertainty)[0]

    wasserstein_distance_sq = wasserstein_distance(squared_error, y_uncertainty)
    wasserstein_distance_abs = wasserstein_distance(absolute_error, y_uncertainty)

    sharpness_sq = np.var(
        y_uncertainty - squared_error
    )  # https://arxiv.org/abs/2206.07795?context=eess.IV
    sharpness_abs = np.var(y_uncertainty - absolute_error)

    # gaussian negative log likelihood
    # https://arxiv.org/abs/1807.09289

    return {
        "pearson_sq": pearson_sq,
        "spearman_sq": spearman_sq,
        "pearson_abs": pearson_abs,
        "spearman_abs": spearman_abs,
        "wasserstein_distance_sq": wasserstein_distance_sq,
        "wasserstein_distance_abs": wasserstein_distance_abs,
        "sharpness_sq": sharpness_sq,
        "sharpness_abs": sharpness_abs,
    }


def plot_uncertainty(
    y_prediction: np.ndarray,
    y_target: np.ndarray,
    y_uncertainty: np.ndarray,
    compare: str = "squared_error",
) -> None:
    """
    plot uncertainty against target values. compare can be "squared_error" or "absolute_error".
    """
    error = y_target - y_prediction
    if compare == "squared_error":
        error = np.square(error)
    elif compare == "absolute_error":
        error = np.abs(error)
    else:
        raise ValueError("compare must be 'squared_error' or 'absolute_error'.")

    plt.plot(error, y_uncertainty, ".", color="black", alpha=0.5)
    plt.xlabel("error")
    plt.ylabel("Uncertainty")
    plt.show()


def plot_error_reduce_data(
    results: pd.DataFrame,
    show_baseline: bool = True,
    label: str = "Uncertainty",
    x_offset: bool = True,
) -> None:
    """Plot the error reduction as a function of the dataset size left using data order wrt. uncertainty and wrt. distance from the mean prediction.

    Args:
        results (pd.DataFrame): dataframe with columns "response", "y_preds", "y_uncertainty"
        show_baseline (bool, optional): Whether to show the of distance to mean prediction. Defaults to True.
        label (str, optional): Label for the uncertainty plot. Defaults to "Uncertainty".
        x_offset (bool, optional): Whether to pad the x-axis. Defaults to True.

    """
    results = results.copy()
    results["sq_error"] = (results["response"] - results["y_preds"]) ** 2
    results_ordered = results.sort_values("y_uncertainty")

    results_ordered["cummulative_sq_error"] = np.cumsum(
        results_ordered["sq_error"]
    ) / range(1, len(results_ordered) + 1)

    results["distance_from_mean"] = np.abs(
        results["y_preds"] - results["y_preds"].mean()
    )
    results_sorted_dist_from_mean = results.sort_values("distance_from_mean")

    results_sorted_dist_from_mean["cummulative_sq_error"] = np.cumsum(
        results_sorted_dist_from_mean["sq_error"]
    ) / range(1, len(results_sorted_dist_from_mean) + 1)

    plt.plot(
        range(len(results_ordered)),
        results_ordered["cummulative_sq_error"],
        ".",
        markersize=0.3,
        label=label,
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Quantile of data")

    plt.gca().yaxis.grid(color="gray", linestyle="dashed", alpha=0.23)
    plt.gca().set_axisbelow(True)
    plt.ylabel("Mean Squared Error")
    if show_baseline:
        plt.plot(
            range(len(results_sorted_dist_from_mean)),
            results_sorted_dist_from_mean["cummulative_sq_error"],
            ".",
            markersize=0.3,
            label="Distance from the mean prediction",
        )
    quantiles = [1, 0.75, 0.5, 0.25, 0.0]
    results_ordered = results_ordered.reset_index()
    data_quantiles = [int(np.quantile(results_ordered.index, q)) for q in quantiles]
    plt.xticks(data_quantiles, quantiles)
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_markersize(10)
    # plt.gca().set_ylim(0, 1)
    if x_offset:
        plt.gca().set_xlim(len(results_ordered) + 1000, -4000)
    plt.show()
