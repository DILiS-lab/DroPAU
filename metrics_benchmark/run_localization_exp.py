import argparse
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
from localization import localization_experiment
from sklearn.preprocessing import StandardScaler


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


@to_serializable.register(np.float64)
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

    for dataset in ["synthetic", "synthetic_mixed_5", "red_wine", "ailerons", "lsat"]:
        if dataset == "synthetic":
            data = get_synthetic_data(
                41500,
                70,
                0,
                n_samples_train=32000,
                n_samples_val=8000,
                n_samples_test=1500,
            )

            for run in [1, 2, 3]:
                # ------------- SIMPLE NOISE MODEL
                print(
                    f"----------------------------------- SIMPLE NOISE MODEL DC {dataset} run {run} -----------------------------------"
                )
                (
                    out_localization_precision,
                    out_localization_mass_accuracy,
                    out_global_metrics,
                ) = localization_experiment(
                    **data,
                    noise_levels=[1, 2],  #
                    n_noise_features_list=[1, 2, 5],  # , 2, 5
                    use_simple_noise_model=True,
                    dataset=f"{dataset}_{run}_simple",
                )
                write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_out_localization_simple_{run}.json",
                )

                # ------------- NORMAL NOISE MODEL
                print(
                    f"----------------------------------- NORMAL NOISE MODEL DC {dataset} run {run} -----------------------------------"
                )
                (
                    out_localization_precision,
                    out_localization_mass_accuracy,
                    out_global_metrics,
                ) = localization_experiment(
                    **data,
                    noise_levels=[1, 2],
                    n_noise_features_list=[1, 2, 5],
                    dataset=f"t_{dataset}_{run}"                )
                write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_out_localization_{run}.json",
                )
        elif("synthetic" in dataset): #mixed data
            n_noise_features_mixed = int(dataset.split("_")[2])
            data = get_synthetic_data(
                    41500,
                    70,
                    0,
                    n_samples_train=32000,
                    n_samples_val=8000,
                    n_samples_test=1500,
                )
            print(
                    f"----------------------------------- NORMAL NOISE MODEL DC {dataset} MIXED -------------------------------"
                )
            (
                out_localization_precision,
                out_localization_mass_accuracy,
                out_global_metrics,
            ) = localization_experiment(
                **data,
                noise_levels=[ 2],
                n_noise_features_list=[5],
                dataset=f"t_{dataset}", n_noise_features_mixed=n_noise_features_mixed,
                 use_simple_noise_model=False )
            write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_out_localization.json",
                )
            print(
                    f"----------------------------------- SIMPLE NOISE MODEL DC {dataset} MIXED -------------------------------"
                )
            (
                out_localization_precision,
                out_localization_mass_accuracy,
                out_global_metrics,
            ) = localization_experiment(
                **data,
                noise_levels=[2],
                n_noise_features_list=[5],
                dataset=f"t_{dataset}", n_noise_features_mixed=n_noise_features_mixed,
                 use_simple_noise_model=True )
            write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_out_localization_simple.json",)

        else:
            for repeat in [1, 50]:
                print(
                    f"----------------------------------- NORMAL NOISE MODEL DC {dataset} repeat {repeat} -----------------------------------"
                )
                if dataset == "red_wine":
                    data = get_red_wine_dataset()
                elif dataset == "ailerons":
                    data = get_ailerons_dataset()
                elif dataset == "lsat":
                    data = get_LSAT_dataset()

                data["x_train"] = np.repeat(data["x_train"], repeat, axis=0)
                data["y_train"] = np.repeat(data["y_train"], repeat, axis=0)

                data = standardize_output(data)
                (
                    out_localization_precision,
                    out_localization_mass_accuracy,
                    out_global_metrics,
                ) = localization_experiment(
                    **data,
                    noise_levels=[1, 2],
                    n_noise_features_list=[1, 2, 5],
                    dataset=dataset,
                )
                write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_{repeat}_out_localization.json",
                )

            print(
                f"----------------------------------- SIMPLE NOISE MODEL DC {dataset} -----------------------------------"
            )
            if dataset == "red_wine":
                data = get_red_wine_dataset()
            elif dataset == "ailerons":
                data = get_ailerons_dataset()
            elif dataset == "lsat":
                data = get_LSAT_dataset()

            data = standardize_output(data)
            (
                out_localization_precision,
                out_localization_mass_accuracy,
                out_global_metrics,
            ) = localization_experiment(
                **data,
                noise_levels=[1, 2],
                n_noise_features_list=[1, 2, 5],
                use_simple_noise_model=True,
                dataset=dataset,
            )
            write_to_json(
                {
                    "local_localization_precision": out_localization_precision,
                    "local_localization_mass_accuracy": out_localization_mass_accuracy,
                    "global_metrics": out_global_metrics,
                },
                f"results/fixed_{dataset}_localization_simple.json",
            )


if __name__ == "__main__":
    main()
    print("Done Localization")