import argparse
import json
from functools import singledispatch
import os
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
from sklearn.utils import shuffle


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


def shuffle_data(data):
    """
    Shuffles the train, test, and validation datasets together and re-splits them
    into new train, test, and validation sets.
    """
    # Combine train, test, and validation datasets
    x_combined = np.concatenate([data["x_train"], data["x_test"], data["x_val"]], axis=0)
    y_combined = np.concatenate([data["y_train"], data["y_test"], data["y_val"]], axis=0)

    # Shuffle the combined data
    x_shuffled, y_shuffled = shuffle(x_combined, y_combined, random_state=42)

    # Recalculate the sizes of train, test, and validation sets
    n_train = len(data["x_train"])
    n_val = len(data["x_val"])

    # Split shuffled data back into train, test, and validation
    data["x_train"], data["x_val"], data["x_test"] = (
        x_shuffled[:n_train],
        x_shuffled[n_train : n_train + n_val],
        x_shuffled[n_train + n_val :],
    )
    data["y_train"], data["y_val"], data["y_test"] = (
        y_shuffled[:n_train],
        y_shuffled[n_train : n_train + n_val],
        y_shuffled[n_train + n_val :],
    )
    return data

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


    parser = argparse.ArgumentParser(description="Run localization experiments with different datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["synthetic", "synthetic_mixed_5", "red_wine", "ailerons", "lsat"],
        help="Specify the dataset to use.",
    )
    args = parser.parse_args()

    dataset = args.dataset 


    torch.set_float32_matmul_precision("medium")
    for run in range(5):
        if dataset == "synthetic":
            if os.path.exists(f"results/fixed_{dataset}_out_localization_{run}.json") and os.path.exists(f"results/fixed_{dataset}_out_localization_simple_{run}.json"):
                continue
            data = get_synthetic_data(
                41500,
                70,
                0,
                n_samples_train=32000,
                n_samples_val=8000,
                n_samples_test=1500,
                random_state=run,
            )

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
                noise_levels=[2],  #
                n_noise_features_list=[5],  # , 2, 5
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
                noise_levels=[2],
                n_noise_features_list=[5],
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
            if os.path.exists(f"results/fixed_{dataset}_out_localization_{run}.json") and os.path.exists(f"results/fixed_{dataset}_out_localization_simple_{run}.json"):
                continue
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
                noise_levels=[2],
                n_noise_features_list=[5],
                dataset=f"t_{dataset}", n_noise_features_mixed=n_noise_features_mixed,
                use_simple_noise_model=False )
            write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_out_localization_{run}.json",
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
                    f"results/fixed_{dataset}_out_localization_simple_{run}.json",)

        else:
            
            for repeat in [50]:
                if os.path.exists(f"results/fixed_{dataset}_{repeat}_out_localization_{run}.json") and os.path.exists(f"results/fixed_{dataset}_localization_simple_{run}.json"):
                    continue
                
                print(
                    f"----------------------------------- NORMAL NOISE MODEL DC {dataset} repeat {repeat} -----------------------------------"
                )
                if dataset == "red_wine":
                    data = get_red_wine_dataset()
                elif dataset == "ailerons":
                    data = get_ailerons_dataset()
                elif dataset == "lsat":
                    data = get_LSAT_dataset()
                data = shuffle_data(data)

                data["x_train"] = np.repeat(data["x_train"], repeat, axis=0)
                data["y_train"] = np.repeat(data["y_train"], repeat, axis=0)
                data = standardize_output(data)
                (
                    out_localization_precision,
                    out_localization_mass_accuracy,
                    out_global_metrics,
                ) = localization_experiment(
                    **data,
                    noise_levels=[2],
                    n_noise_features_list=[5],
                    dataset=dataset,
                )
                write_to_json(
                    {
                        "local_localization_precision": out_localization_precision,
                        "local_localization_mass_accuracy": out_localization_mass_accuracy,
                        "global_metrics": out_global_metrics,
                    },
                    f"results/fixed_{dataset}_{repeat}_out_localization_{run}.json",
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
            data = shuffle_data(data)
            data = standardize_output(data)

            (
                out_localization_precision,
                out_localization_mass_accuracy,
                out_global_metrics,
            ) = localization_experiment(
                **data,
                noise_levels=[2],
                n_noise_features_list=[5],
                use_simple_noise_model=True,
                dataset=dataset,
            )
            write_to_json(
                {
                    "local_localization_precision": out_localization_precision,
                    "local_localization_mass_accuracy": out_localization_mass_accuracy,
                    "global_metrics": out_global_metrics,
                },
                f"results/fixed_{dataset}_localization_simple_{run}.json",
            )


if __name__ == "__main__":
    main()
    print("Done Localization")