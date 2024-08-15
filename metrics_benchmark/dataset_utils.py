import inspect
import os
import sys
from pathlib import Path

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pandas as pd
import requests
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

from utils.experiment_utils import linear_model, noise_model


def get_split(X, y, train_size=None, val_size=None, test_size=None):
    if train_size is not None and val_size is not None and test_size is not None:
        assert train_size + val_size + test_size == len(X)

    train_size = train_size if train_size is not None else np.floor(0.7 * len(X))
    val_size = val_size if val_size is not None else np.floor(0.1 * len(X))

    # theoretically unnercessary if data is synthetic but doesn't hurt
    random_idx = np.random.RandomState(seed=0).permutation(len(X))

    train_idx = random_idx[: int(train_size)]
    val_idx = random_idx[int(train_size) : int(train_size + val_size)]
    test_idx = random_idx[int(train_size + val_size) :]

    X_train = X.iloc[train_idx].to_numpy()
    y_train = y.iloc[train_idx].to_numpy()

    X_val = X.iloc[val_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()

    X_test = X.iloc[test_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_val": X_val,
        "y_val": y_val,
        "x_test": X_test,
        "y_test": y_test,
    }


def get_LSAT_dataset():
    Path("data").mkdir(parents=True, exist_ok=True)
    if not Path("data/law_school_cf_train.csv").is_file():
        print("Downloading LSAT dataset")
        download_url = "https://raw.githubusercontent.com/throwaway20190523/MonotonicFairness/master/data/law_school_cf_train.csv"
        with open("data/law_school_cf_train.csv", mode="wb") as file:
            file.write(requests.get(download_url).content)

    if not Path("data/law_school_cf_test.csv").is_file():
        download_url = "https://raw.githubusercontent.com/throwaway20190523/MonotonicFairness/master/data/law_school_cf_test.csv"
        with open("data/law_school_cf_test.csv", mode="wb") as file:
            file.write(requests.get(download_url).content)

    df_train = pd.read_csv("data/law_school_cf_train.csv")
    df_test = pd.read_csv("data/law_school_cf_test.csv")

    val_size = np.floor(0.1 * len(df_train))

    random_idx = np.random.RandomState(seed=0).permutation(len(df_train))

    val_idx = random_idx[: int(val_size)]
    train_idx = random_idx[int(val_size) :]

    X_train = df_train.drop(columns=["ZFYA"]).iloc[train_idx].to_numpy()
    y_train = df_train["ZFYA"].iloc[train_idx].to_numpy()

    X_val = df_train.drop(columns=["ZFYA"]).iloc[val_idx].to_numpy()
    y_val = df_train["ZFYA"].iloc[val_idx].to_numpy()

    X_test = df_test.drop(columns=["ZFYA"]).to_numpy()
    y_test = df_test["ZFYA"].to_numpy()

    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_val": X_val,
        "y_val": y_val,
        "x_test": X_test,
        "y_test": y_test,
    }


def get_ailerons_dataset():
    temp = fetch_openml(data_id="296", parser="auto", as_frame=True)
    X = temp.data
    y = temp.target
    return get_split(X, y)


def get_red_wine_dataset():
    wine_quality = fetch_ucirepo(id=186)

    red_wine = wine_quality["data"]["original"].query("color == 'red'")

    X = red_wine.drop(columns=["quality", "color"])
    y = red_wine["quality"]

    return get_split(X, y)


def get_synthetic_data(
    n_samples,
    n_features_mean,
    n_features_uncertainty,
    n_features_mean_and_uncertainty: int = 0,
    noise_scaler: int = 1,
    random_state: int = 0,
    n_samples_train: int = None,
    n_samples_val: int = None,
    n_samples_test: int = None,
):
    from sklearn.preprocessing import StandardScaler

    model_error_linear = 0.02
    model_error_noise = 0.05

    # generate the mean features and mean labels
    data = linear_model(
        n_samples, n_features_mean, noise=model_error_linear, random_state=random_state
    )
    inputs = data[0]
    output = data[1]
    if n_features_uncertainty > 0:
        # generate the uncertainty features and uncertainty labels
        data_noise = noise_model(n_samples, n_features_uncertainty, model_error_noise)

        # generate the heteroscedastic noise based on output of the noise model
        noise = np.random.normal(
            loc=0.0, scale=noise_scaler * data_noise[1], size=n_samples
        )
        output = output + noise
        feature_names = [f"feature_{i}" for i in range(inputs.shape[1])] + [
            f"noise_feature_{i}" for i in range(data_noise[0].shape[1])
        ]
        inputs = np.concatenate((inputs, data_noise[0]), axis=1)
    else:
        feature_names = [f"feature_{i}" for i in range(inputs.shape[1])]
    if n_features_mean_and_uncertainty > 0:
        # generate the mean features and mean labels
        data_mixed = noise_model(
            n_samples, n_features_mean_and_uncertainty, model_error_noise
        )
        # standardize the data
        location_change = (data_mixed[1]-np.mean(data_mixed[1]))/np.std(data_mixed[1])
        # scale to the same range as the original data for meaningful contribution
        shift = np.random.normal(loc=location_change*np.std(output), scale=2*noise_scaler*data_mixed[1], size=n_samples)
        inputs = np.concatenate((inputs, data_mixed[0]), axis=1)
        output = output + shift
        feature_names += [f"mixed_feature_{i}" for i in range(data_mixed[0].shape[1])]
        

    splits = get_split(
        pd.DataFrame(inputs, columns=feature_names),
        pd.Series(output),
        n_samples_train,
        n_samples_val,
        n_samples_test,
    )

    splits["y_train"] = splits["y_train"].reshape(-1, 1)
    splits["y_val"] = splits["y_val"].reshape(-1, 1)
    splits["y_test"] = splits["y_test"].reshape(-1, 1)
    scaler = StandardScaler()
    splits["y_train"] = scaler.fit_transform(splits["y_train"])
    splits["y_val"] = scaler.transform(splits["y_val"])
    splits["y_test"] = scaler.transform(splits["y_test"])
    splits["y_train"] = splits["y_train"].flatten()
    splits["y_val"] = splits["y_val"].flatten()
    splits["y_test"] = splits["y_test"].flatten()

    return splits


if __name__ == "__main__":
    a = get_ailerons_dataset()
    a_meta = {
        "Train_size": len(a["x_train"]),
        "Val_size": len(a["x_val"]),
        "Test_size": len(a["x_test"]),
        "features": a["x_train"].shape[1],
    }
    print(f"Ailerons: {a_meta}")

    lsat = get_LSAT_dataset()
    lsat_meta = {
        "Train_size": len(lsat["x_train"]),
        "Val_size": len(lsat["x_val"]),
        "Test_size": len(lsat["x_test"]),
        "features": lsat["x_train"].shape[1],
    }
    print(f"LSAT: {lsat_meta}")

    rw = get_red_wine_dataset()
    rw_meta = {
        "Train_size": len(rw["x_train"]),
        "Val_size": len(rw["x_val"]),
        "Test_size": len(rw["x_test"]),
        "features": rw["x_train"].shape[1],
    }
    print(f"Red Wine: {rw_meta}")
