import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from utils import get_labels, get_true_data, load_pulsar, load_intention
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import os

# datafile = 'output/exp1_pulsar_data.json'
RANDOM_STATE = 1
plot_dir = os.path.join("plots", "part1")


def get_stats(data):
    xs = data.keys()

    silhouette = [data[i]["silhouette"] for i in xs]
    sse_scores = -np.array([data[i]["score"] for i in xs])
    avg_entropy = [data[i]["average_entropy"] for i in xs]
    homogeneity = [data[i]["homogeneity"] for i in xs]
    completeness = [data[i]["completeness"] for i in xs]
    v_measure = [data[i]["v_measure"] for i in xs]

    return silhouette, sse_scores, avg_entropy, homogeneity, completeness, v_measure


def save_clustering_plots(datafile, prefix, plot_dir="plots/part1"):
    with open(datafile, "r") as f:
        data = json.load(f)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.001)

    kmeans_data = {k: v for k, v in data["kmeans"].items()}
    em_data = {k: v for k, v in data["em"].items()}

    xs = list(kmeans_data.keys())
    kmeans_silhoutte, kmeans_sse, kmeans_avg_entropy, _, _, _ = get_stats(kmeans_data)
    em_silhouette, em_sse, em_avg_entropy, _, _, _ = get_stats(em_data)

    (kmeans,) = ax1.plot(xs, kmeans_silhoutte)
    ax2.plot(xs, kmeans_sse)
    ax3.plot(xs, kmeans_avg_entropy)

    ax2_em = ax2.twinx()
    (em,) = ax1.plot(xs, em_silhouette, c="green")
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
    outpath = os.path.join(plot_dir, f"{prefix}_ClusterSearch.png")
    plt.savefig(outpath)
    plt.close()


def save_cluster_evaluations_plots(datafile, prefix, plot_dir):
    with open(datafile, "r") as f:
        data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.001)

    kmeans_data = {k: v for k, v in data["kmeans"].items()}
    em_data = {k: v for k, v in data["em"].items()}

    xs = list(kmeans_data.keys())
    _, _, _, kh, kc, kv = get_stats(kmeans_data)
    _, _, _, eh, ec, ev = get_stats(em_data)

    (kmeans,) = ax1.plot(xs, kh)
    ax2.plot(xs, kc)
    ax3.plot(xs, kv)

    (em,) = ax1.plot(xs, eh, c="green")
    ax2.plot(xs, ec, c="green")
    ax3.plot(xs, ev, c="green")

    ax3.set_xlabel("# of Clusters")
    ax1.set_ylabel("Homogeneity")
    ax2.set_ylabel("Completeness")
    ax3.set_ylabel("V-Measure")

    ax1.set_xlim((0, 28))
    ax1.legend((kmeans, em), ("K-Means", "EM"))

    # for ax in [ax1, ax2, ax3]:
    #     ax.set_xlim((2, 30))
    ax1.set_title("Cluster Search")
    outpath = os.path.join(plot_dir, f"{prefix}_ClusterEvaluation.png")
    plt.savefig(outpath)
    plt.close()


def print_evaluation_stats(datafile, kmeans_clusters=2, em_clusters=2):
    with open(datafile, "r") as f:
        data = json.load(f)

    kmeans_data = {k: v for k, v in data["kmeans"].items()}
    em_data = {k: v for k, v in data["em"].items()}

    for clusterer, data, clusters in zip(["K-Means", "EM"], [kmeans_data, em_data], [kmeans_clusters, em_clusters]):
        print(f"Evaluation Stats for Clusterer {clusterer} with {clusters} clusters")
        _, _, _, h, c, v = get_stats(data)
        print(f"Homogeneity: {h[clusters]:.4f}")
        print(f"Completeness: {c[clusters]:.4f}")
        print(f"V-Measure: {v[clusters]:.4f}")


def print_clustering_stats(X, y):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.001)

    kmeans = KMeans(2, random_state=RANDOM_STATE)
    em = GaussianMixture(2, random_state=RANDOM_STATE)

    kmeans.fit(X)
    em.fit(X)

    preds_kmeans = kmeans.predict(X)
    preds_em = em.predict(X)

    best_accuracy_em = max(accuracy_score(y, preds_em), accuracy_score(y, 1 - preds_em))
    best_accuracy_kmeans = max(
        accuracy_score(y, preds_kmeans), accuracy_score(y, 1 - preds_kmeans)
    )

    best_f1_score_em = max(f1_score(y, preds_em), f1_score(y, 1 - preds_em))
    best_f1_score_kmeans = max(f1_score(y, preds_kmeans), f1_score(y, 1 - preds_kmeans))

    print(f"Stats assuming clusters are label predictions")
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
    plot_dir = os.path.join("plots", "part1")
    intention_datafile = os.path.join("output", "part1", "exp1_intention_data.json")
    pulsar_datafile = os.path.join("output", "part1", "exp1_pulsar_data.json")
    save_clustering_plots(intention_datafile, "intention")
    save_clustering_plots(pulsar_datafile, "pulsar")
    save_cluster_evaluations_plots(intention_datafile, "intention", plot_dir)
    save_cluster_evaluations_plots(pulsar_datafile, "pulsar", plot_dir)
    print("Intention Evaluation")
    print_evaluation_stats(intention_datafile)

    print("Pulsar Evaluation")
    print_evaluation_stats(pulsar_datafile)

    # intention_X, intention_y = load_intention()
    # pulsar_X, pulsar_y = load_pulsar()
    # print_clustering_stats(intention_X, intention_y)
    # print_clustering_stats(pulsar_X, pulsar_y)
