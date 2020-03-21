import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from utils import get_labels, get_true_data
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import os

# datafile = 'output/exp1_pulsar_data.json'
RANDOM_STATE = 1
plot_dir = os.path.join("plots", "part1")


def get_stats(data):
    xs = data.keys()

    silhoutte_scores = [data[i]["silhouette"] for i in xs]
    sse_scores = -np.array([data[i]["score"] for i in xs])
    avg_entropy = [data[i]["average_entropy"] for i in xs]

    return silhoutte_scores, sse_scores, avg_entropy


def save_clustering_plots(dataset):
    datafile = f"output/exp1_{dataset}_data.json"
    with open(datafile, "r") as f:
        data = json.load(f)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.001)

    kmeans_data = {k: v for k, v in data["kmeans"].items()}
    em_data = {k: v for k, v in data["em"].items()}

    xs = list(kmeans_data.keys())
    kmeans_silhoutte, kmeans_sse, kmeans_avg_entropy = get_stats(kmeans_data)
    em_silhoutte, em_sse, em_avg_entropy = get_stats(em_data)

    (kmeans,) = ax1.plot(xs, kmeans_silhoutte)
    ax2.plot(xs, kmeans_sse)
    ax3.plot(xs, kmeans_avg_entropy)

    ax2_em = ax2.twinx()
    (em,) = ax1.plot(xs, em_silhoutte, c="green")
    ax2_em.plot(xs, em_sse, c="green")
    ax3.plot(xs, em_avg_entropy, c="green")

    ax3.set_xlabel("# of Clusters")
    ax1.set_ylabel("Silhouette Score")
    ax2.set_ylabel("SSE")
    ax2_em.set_ylabel("Log-Likelihood")
    ax3.set_ylabel("Weighted Average Entropy")

    ax1.set_xlim((0, 28))
    ax1.legend((kmeans, em), ("K-Means", "EM"))

    # for ax in [ax1, ax2, ax3]:
    #     ax.set_xlim((2, 30))
    ax1.set_title("Cluster Search")
    outpath = os.path.join(plot_dir, f"{dataset}_Part1ClusterSearch.png")
    plt.savefig(outpath)
    plt.close()


def print_clustering_stats(dataset):
    datafile = f"output/part1/exp1_{dataset}_data.json"
    with open(datafile, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.001)

    kmeans_data = {k: v for k, v in data["kmeans"].items()}
    em_data = {k: v for k, v in data["em"].items()}

    true_y = get_labels(dataset)
    X = get_true_data(dataset)

    kmeans = KMeans(2, random_state=RANDOM_STATE)
    em = GaussianMixture(2, random_state=RANDOM_STATE)

    kmeans.fit(X)
    em.fit(X)

    preds_kmeans = kmeans.predict(X)
    preds_em = em.predict(X)

    best_accuracy_em = max(
        accuracy_score(true_y, preds_em), accuracy_score(true_y, 1 - preds_em)
    )
    best_accuracy_kmeans = max(
        accuracy_score(true_y, preds_kmeans), accuracy_score(true_y, 1 - preds_kmeans)
    )

    best_f1_score_em = max(f1_score(true_y, preds_em), f1_score(true_y, 1 - preds_em))
    best_f1_score_kmeans = max(
        f1_score(true_y, preds_kmeans), f1_score(true_y, 1 - preds_kmeans)
    )

    print(f"Stats for {dataset} dataset assuming clusters are label predictions")
    print("---------------")
    print("K-Means Results")
    print(f"Accuracy: {best_accuracy_kmeans:.4f}")
    print(f"F1 Score: {best_f1_score_kmeans:.4f}")
    print("---------------")
    print("EM Results")
    print(f"Accuracy: {best_accuracy_em:.4f}")
    print(f"F1 Score: {best_f1_score_em:.4f}")
    print()


if __name__ == "__main__":
    save_clustering_plots("intention")
    save_clustering_plots("pulsar")
    print_clustering_stats("intention")
    print_clustering_stats("pulsar")
