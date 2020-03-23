import json
from clustering import run_clustering_experiment
from dimensionality_reduction import save_transformed_data, save_all_transformations
from utils import (
    load_intention,
    load_intention_PCA_reduced,
    load_intention_ICA_reduced,
    load_intention_LLE_reduced,
    load_intention_RP_reduced,
    load_pulsar,
    load_pulsar_PCA_reduced,
    load_pulsar_ICA_reduced,
    load_pulsar_RP_reduced,
    load_pulsar_LLE_reduced,
)
import os
from nn import tune_part5networks, tune_part4networks, get_all_evaluations, print_best_params, print_best_params_part5, evaluate_part5networks

OUTPUT_DIR = "output"


def part1():
    part1_dir = os.path.join(OUTPUT_DIR, "part1")
    X1, y1 = load_pulsar()
    X2, y2 = load_intention()

    print("Running Exp1 on Pulsar Dataset")
    pulsar_results = run_clustering_experiment(X1, y1)
    pulsar_filename = os.path.join(part1_dir, "exp1_pulsar_data.json")

    with open(pulsar_filename, "w") as f:
        json.dump(pulsar_results, f)

    print("Running Exp1 on Intention Dataset")
    intention_results = run_clustering_experiment(X2, y2)
    intention_filename = os.path.join(part1_dir, "exp1_intention_data.json")

    with open(intention_filename, "w") as f:
        json.dump(intention_results, f)


def part2(intention=True, pulsar=True):
    part2_dir = os.path.join(OUTPUT_DIR, "part2")
    if intention:
        X, y = load_intention()
        print("Collecting Exp2 on Intention")
        save_all_transformations(X, "intention", output_dir=part2_dir)

    if pulsar:
        X, y = load_pulsar()
        print("Collecting Exp2 Data on Pulsar")
        save_all_transformations(
            X,
            "pulsar",
            output_dir=part2_dir,
            ica_max_component_ratio=1.0,
            rp_component_ratio=1.0,
            lle_max_component_ratio=1.0,
        )


# TODO: Modify part 3 to use reduced data and pass to clusterer
def part3(intention=True, pulsar=True, transformers=["PCA", "ICA", "RP", "LLE"]):
    part3_dir = os.path.join(OUTPUT_DIR, "part3")

    if intention:
        if "PCA" in transformers:
            X, y = load_intention_PCA_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "intention_pca_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

        if "ICA" in transformers:
            X, y = load_intention_ICA_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "intention_ica_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

        if "RP" in transformers:
            X, y = load_intention_RP_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "intention_rp_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

        if "LLE" in transformers:
            X, y = load_intention_LLE_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "intention_lle_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

    if pulsar:
        if "PCA" in transformers:
            X, y = load_pulsar_PCA_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "pulsar_pca_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

        if "ICA" in transformers:
            X, y = load_pulsar_ICA_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "pulsar_ica_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

        if "RP" in transformers:
            X, y = load_pulsar_RP_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "pulsar_rp_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)

        if "LLE" in transformers:
            X, y = load_pulsar_LLE_reduced()
            results = run_clustering_experiment(X, y)
            output = os.path.join(part3_dir, "pulsar_lle_clustering.json")
            with open(output, "w") as f:
                json.dump(results, f)


def part4():
    part4_dir = os.path.join(OUTPUT_DIR, "part4")
    tune_part4networks()
    results = get_all_evaluations()
    print_best_params()
    print(results)


def part5():
    part5_dir = os.path.join(OUTPUT_DIR, "part5")
    tune_part5networks()
    print_best_params_part5()
    results = evaluate_part5networks()
    print(results)


if __name__ == "__main__":
    part1()
    part2(intention=True, pulsar=True)
    part3(intention=True, pulsar=True)
    part4()
    part5()
