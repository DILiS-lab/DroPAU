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
from perturbation import perturbation_experiment
from sklearn.preprocessing import StandardScaler


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


def main():
    torch.set_float32_matmul_precision("medium")

    for dataset in ["synthetic_mixed_5"]: #["synthetic", "red_wine", "ailerons", "lsat"]:
        if 'synthetic' in dataset:
            data = get_synthetic_data(
                41500,
                70,
                5,
                0 if dataset == "synthetic" else int(dataset.split("_")[2]),
                n_samples_train=32000,
                n_samples_val=8000,
                n_samples_test=1500,
            )
            out_perturb_pos = perturbation_experiment(
                **data, epsilons=[0.01, 0.05], direction="positive", top_k=4, warn=False
            )
            out_perturb_neg = perturbation_experiment(
                **data, epsilons=[0.01, 0.05], direction="negative", top_k=4
            )
            write_to_json(
                {"perturb_pos": out_perturb_pos, "perturb_neg": out_perturb_neg},
                "synthetic_out_perturbation.json",
            )

        else:
            for repeat in [1, 100]:
                if dataset == "red_wine":
                    data = get_red_wine_dataset()
                elif dataset == "ailerons":
                    data = get_ailerons_dataset()
                elif dataset == "lsat":
                    data = get_LSAT_dataset()

                data["x_train"] = np.repeat(data["x_train"], repeat, axis=0)
                data["y_train"] = np.repeat(data["y_train"], repeat, axis=0)

                data = standardize_output(data)
                out_perturb_pos = perturbation_experiment(
                    **data, epsilons=[0.01, 0.05], direction="positive", top_k=4
                )
                out_perturb_neg = perturbation_experiment(
                    **data, epsilons=[0.01, 0.05], direction="negative", top_k=4
                )
                write_to_json(
                    {"perturb_pos": out_perturb_pos, "perturb_neg": out_perturb_neg},
                    f"{dataset}_{repeat}_out_perturbation.json",
                )


if __name__ == "__main__":
    main()
