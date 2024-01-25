import numpy as np
import pandas as pd
import xgboost as xgb


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
    model, x_val, y_val, x_test, variable_names, sort=True, return_model=False
):
    """
    Explain the model using InfoBoost
    """
    error = y_val - model.predict(x_val)

    error_model = train_xgboost_var(error, x_val)

    uncertainty = error_model.predict(x_test)
    x_test_sorted = x_test[np.argsort(uncertainty, kind="stable")]

    shapley_values = pd.DataFrame(
        error_model.get_booster().predict(
            xgb.DMatrix(x_test_sorted if sort else x_test), pred_contribs=True
        ),
        columns=variable_names,
    )
    shapley_values = shapley_values.drop(columns=["bias"])

    abs_shapley_values = shapley_values.abs()
    if return_model:
        return {
            "shapley_values": shapley_values.values,
            "feature_importances": abs_shapley_values.values,
            "instances_to_explain": x_test_sorted if sort else x_test,
            "error_model": error_model,
        }
    else:
        return {
            "shapley_values": shapley_values.values,
            "feature_importances": abs_shapley_values.values,
            "instances_to_explain": x_test_sorted if sort else x_test,
        }
