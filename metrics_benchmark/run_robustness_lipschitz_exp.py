import json
from functools import singledispatch

import numpy as np
import torch
from dataset_utils import (
    get_ailerons_dataset,
    get_LSAT_dataset,
    get_red_wine_dataset,
    get_synthetic_data,
)
from robustness_lipschitz import robustness_experiment
from sklearn.preprocessing import StandardScaler
import argparse


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


# Standardization for non synthetic datasets
def standardize_output(data):
    data["y_train"] = data["y_train"].reshape(-1, 1)
    data["y_test"] = data["y_test"].reshape(-1, 1)
    data["y_val"] = data["y_val"].reshape(-1, 1)

    scaler = StandardScaler()
    data["y_train"] = scaler.fit_transform(data["y_train"]).flatten()
    data["y_test"] = scaler.transform(data["y_test"]).flatten()
    data["y_val"] = scaler.transform(data["y_val"]).flatten()

    return data


def write_to_json(object, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(object, f, ensure_ascii=False, indent=4, default=to_serializable)


def get_parser():
    parser = argparse.ArgumentParser(description="Robustness experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Name of dataset",
    )

    return parser


def main():
    torch.set_float32_matmul_precision("medium")

    parser = get_parser()
    user_args = parser.parse_args()

    for dataset in [user_args.dataset]:  # "lsat", "ailerons", "red_wine", "synthetic"
        if dataset == "synthetic":
            data = get_synthetic_data(
                41500,
                70,
                5,
                n_samples_train=32000,
                n_samples_val=8000,
                n_samples_test=1500,
            )

            out_robust = robustness_experiment(**data, dataset=f"{dataset}_fixed")
            write_to_json(out_robust, f"results/{dataset}_out_lipschitz_fixed.json")

        else:
            if dataset == "red_wine":
                data = get_red_wine_dataset()
            elif dataset == "ailerons":
                data = get_ailerons_dataset()
            elif dataset == "lsat":
                data = get_LSAT_dataset()

            data = standardize_output(data)
            out_robust = robustness_experiment(**data, dataset=f"{dataset}_fixed")
            write_to_json(
                out_robust,
                f"results/{dataset}_out_lipschitz_fixed.json",
            )


if __name__ == "__main__":
    main()
