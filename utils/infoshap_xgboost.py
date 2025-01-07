import numpy as np
import pandas as pd
import xgboost as xgb
import os


def train_xgboost(x_train, y_train):
    """
    Train the InfoBoost model
    """
    model = xgb.XGBRegressor(nrounds=25, verbosity=0)
    model.fit(x_train, y_train)

    return model


def train_xgboost_var(error, x_val):
    """
    Train the InfoBoost model for the variance
    """
    model = xgb.XGBRegressor(nrounds=25, verbosity=0)
    model.fit(x_val, error)

    return model


def infoboost_explain(
    model,
    x_val,
    y_val,
    instances_to_explain,
    var_names,
    identifier=None,
    save_dir=None,
    save=True,
    sort=False,
    return_model=False,
):
    """
    Explain the model using InfoBoost
    """
    error = np.log(np.square(y_val - model.predict(x_val)))

    error_model = train_xgboost_var(error, x_val)

    uncertainty = error_model.predict(instances_to_explain)
    instances_to_explain_sorted = instances_to_explain[
        np.argsort(uncertainty, kind="stable")
    ]

    shapley_values = pd.DataFrame(
        error_model.get_booster().predict(
            xgb.DMatrix(instances_to_explain_sorted if sort else instances_to_explain),
            pred_contribs=True,
        ),
        columns=var_names,
    )
    attribution = shapley_values.drop(columns=["bias"])

    feature_importances = shapley_values.abs()
    output = {
            "feature_importance": feature_importances.values,
            "importance_directed": attribution.values,
            "var_names": shapley_values.columns,
            "instances_to_explain": instances_to_explain_sorted
            if sort
            else instances_to_explain,
        }

    if save:
        if not (os.path.exists(f"{save_dir}/importances")):
            os.makedirs(f"{save_dir}/importances")

        np.save(
            f"{save_dir}/importances/infoshap_importances_{identifier}.npy",
            output,
        )
    else:
        if return_model:
            output["error_model"] = error_model
        return output
        
