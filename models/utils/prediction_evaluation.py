import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt


def prediction_quality_metrics(y_prediction: np.ndarray, y_target: np.ndarray) -> dict:
    """
    Calculates prediction quality metrics for the given prediction and target values.
    """
    mse = mean_squared_error(y_target, y_prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_target, y_prediction)
    pearson = pearsonr(y_target, y_prediction)[0]
    spearman = spearmanr(y_target, y_prediction)[0]

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
    }


def plot_predictions(y_prediction: np.ndarray, y_target: np.ndarray) -> None:
    plt.plot(y_target, y_prediction, ".", color="black", alpha=0.5)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.show()


def plot_predictions_with_uncertainty(y_prediction, y_target, y_uncertainty):
    # maybe use PCA first component of gene expression and interpolate?
    raise NotImplementedError
