import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.experiment_utils import run_uncertainty_explanation_experiment


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for run_uncertainty_explanation_experiment function"
    )

    parser.add_argument(
        "--n_instances_to_explain",
        type=int,
        default=200,
        help="Number of instances to explain (default: 200)",
    )
    parser.add_argument(
        "--explainer_repeats",
        type=int,
        default=1,
        help="Number times the explanation is calculated for error bars and run time benchmarks (default: 5)",
    )
    parser.add_argument(
        "--noise_scaler",
        type=float,
        default=2.0,
        help="Noise scaler value (default: 2.0)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=40000,
        help="number of train + early stopping val datapoints (default: 40000)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1500,
        help="number of test datapoints (default: 1500)",
    )
    parser.add_argument(
        "--remake_data", action="store_true", help="Resample the data if specified"
    )

    parser.add_argument(
        "--beta_gaussian", action="store_true", help="Use beta gaussian loss"
    )


    parser.add_argument(
        "--k_mixed",
        type=int,
        default=0,
        help="Number of mixed samples to use in the explanation (default: 0)",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=0,
        help="Run ID (default: 0)")
    
    args = parser.parse_args()

    k_mixed = args.k_mixed
    print("Running uncertainty explanation experiment with k_mixed={}".format(k_mixed))  
    run_uncertainty_explanation_experiment(
        n_instances_to_explain=args.n_instances_to_explain,
        explainer_repeats=args.explainer_repeats,
        noise_scaler=args.noise_scaler,
        k_mixed=k_mixed,
        n=args.n,
        n_test=args.n_test,
        remake_data=args.remake_data,
        beta_gaussian=args.beta_gaussian,
        run_id=args.run_id
    )

    print("Done!")


if __name__ == "__main__":
    main()
