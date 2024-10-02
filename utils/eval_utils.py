import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats

def create_error_reduce_data(results: pd.DataFrame, save_dir: str, with_baseline=True, baseline="distance_mean", with_gt_noise_std=True):

    results["sq_error"] = (results["response"] - results["y_preds"]) ** 2
    results_ordered = results.sort_values("y_uncertainty")

    results_ordered["cummulative_sq_error"] = np.cumsum(
        results_ordered["sq_error"]
    ) / range(1, len(results_ordered) + 1)

    results["distance_from_mean"] = np.abs(
        results["y_preds"] - results["y_preds"].mean()
    )
    
    if baseline == "pred_value":
        results_sorted_baseline = results.sort_values("y_preds")

        results_sorted_baseline["cummulative_sq_error"] = np.cumsum(
            results_sorted_baseline["sq_error"]
        ) / range(1, len(results_sorted_baseline) + 1)
    elif baseline == "distance_mean":
        results["distance_from_mean"] = np.abs(
        results["y_preds"] - results["y_preds"].mean()
        )
        results_sorted_baseline = results.sort_values("distance_from_mean")
        results_sorted_baseline["cummulative_sq_error"] = np.cumsum(
        results_sorted_baseline["sq_error"]
    ) / range(1, len(results_sorted_baseline) + 1)
        
    else:
        raise ValueError("baseline must be 'pred_value' or 'distance_mean'")
  
    if with_gt_noise_std:    
        results_sorted_gt_noise_std = results.sort_values("noise_std_test")
        results_sorted_gt_noise_std["cummulative_sq_error"] = np.cumsum(
            results_sorted_gt_noise_std["sq_error"]
        ) / range(1, len(results_sorted_gt_noise_std) + 1)
    


    pd.DataFrame(results_ordered).to_csv(f'{save_dir}_uncertainty.csv')


    if with_baseline:
        pd.DataFrame(results_sorted_baseline).to_csv(f'{save_dir}_baseline.csv')
    if with_gt_noise_std:
        pd.DataFrame(results_sorted_gt_noise_std).to_csv(f'{save_dir}_ground_truth_std.csv')
       

def create_calibration_data(results, save_dir):
    pms = np.linspace(0, 1, 21)
    fx_y = (results["response"]- results["y_preds"])/np.sqrt(results["y_uncertainty"])
    cdf_fxy= stats.norm(loc=0, scale=1).cdf(fx_y)
    empirical_frequency = [np.mean(cdf_fxy < pm) for pm in pms]

    pd.DataFrame({"pms": pms, "empirical_frequency": empirical_frequency}).to_csv(f'{save_dir}_calibration_data.csv')

