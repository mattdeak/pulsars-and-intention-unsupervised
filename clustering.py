import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils import load_intention, load_pulsar
from sklearn.metrics import silhouette_score
from collections import defaultdict
from scipy.stats import entropy
import json
import os

OUTPUT_DIR = "output"
RANDOM_STATE = 1



def get_weighted_average_entropy(y, cluster_indices):
    clusters = np.unique(cluster_indices)
    N = len(clusters)
    avg_entropy = 0
    for cluster in clusters:
        y_cluster = y.iloc[cluster_indices == cluster]
        weight = y_cluster.size / y.size
        one_probability = y_cluster.sum() / y_cluster.count()
        zero_probability = (y_cluster.count() - y_cluster.sum()) / y_cluster.count()
        avg_entropy += entropy([zero_probability, one_probability]) * weight
    return avg_entropy


def run_clustering_experiment(X, y):
    results = {"kmeans": defaultdict(dict), "em": defaultdict(dict)}
    for i in range(2, 31):
        print(f"Getting scores for {i} clusters")

        kmeans = KMeans(n_clusters=i, random_state=RANDOM_STATE)
        em = GaussianMixture(n_components=i, random_state=RANDOM_STATE)

        kmeans_preds = kmeans.fit_predict(X)
        em_preds = em.fit_predict(X)

        kmeans_silhouette = silhouette_score(X, kmeans_preds)
        em_silhouette = silhouette_score(X, em_preds)

        results["kmeans"][i]["silhouette"] = kmeans_silhouette
        results["em"][i]["silhouette"] = em_silhouette

        kmeans_avg_entropy = get_weighted_average_entropy(y, kmeans_preds)
        em_avg_entropy = get_weighted_average_entropy(y, em_preds)

        results["kmeans"][i]["average_entropy"] = kmeans_avg_entropy
        results["em"][i]["average_entropy"] = em_avg_entropy

        kmeans_score = kmeans.score(X)
        em_score = em.score(X)

        results["kmeans"][i]["score"] = kmeans_score
        results["em"][i]["score"] = em_score

    return results


def run_experiment_1():
    X1, y1 = load_pulsar()
    X2, y2 = load_intention()

    print("Running Exp1 on Pulsar Dataset")
    pulsar_results = run_clustering_experiment(X1, y1)
    pulsar_filename = os.path.join(OUTPUT_DIR, "exp1_pulsar_data.json")

    with open(pulsar_filename, "w") as f:
        json.dump(pulsar_results, f)

    print("Running Exp1 on Intention Dataset")
    intention_results = run_clustering_experiment(X2, y2)
    intention_filename = os.path.join(OUTPUT_DIR, "exp1_intention_data.json")

    with open(intention_filename, "w") as f:
        json.dump(intention_results, f)


if __name__ == "__main__":
    run_experiment_1()
