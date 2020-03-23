import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from utils import get_labels, get_true_data, load_pulsar, load_intention
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import os
from sklearn.manifold import TSNE

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
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.001)

    kmeans_data = {k: v for k, v in data["kmeans"].items()}
    em_data = {k: v for k, v in data["em"].items()}

    xs = list(kmeans_data.keys())
    kmeans_silhoutte, kmeans_sse, kmeans_avg_entropy, _, _, _ = get_stats(kmeans_data)
    em_silhouette, em_sse, em_avg_entropy, _, _, _ = get_stats(em_data)

    (kmeans,) = ax1.plot(xs, kmeans_silhoutte)

    (em,) = ax1.plot(xs, em_silhouette, c="green")

    ax1.set_xlabel("# of Clusters")
    ax1.set_ylabel("Silhouette Score")

    ax1.set_xlim((0, 28))
    ax1.legend((kmeans, em), ("K-Means", "EM"))

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

    for clusterer, data, clusters in zip(
        ["K-Means", "EM"], [kmeans_data, em_data], [kmeans_clusters, em_clusters]
    ):
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


def save_cluster_descriptions(
    data_loader, output_dir, file_prefix, kmeans_clusters=2, em_clusters=2
):
    kmeans = KMeans(kmeans_clusters, random_state=1)
    em = GaussianMixture(n_components=em_clusters, random_state=1)

    tsne2 = TSNE(n_components=2)

    X, y = data_loader()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    kmeans.fit(X)
    em.fit(X)

    reduced = tsne2.fit_transform(X)

    kmeans_clusters = kmeans.predict(X)
    em_probs = em.predict_proba(X)[:, 0]

    class0_ix = y == 0
    class1_ix = y == 1

    class0_reduced = reduced[class0_ix]
    class0_y = y[class0_ix]
    kmeans_clusters_class0 = kmeans_clusters[class0_ix]
    em_probs_class0 = em_probs[class0_ix]
    class1_reduced = reduced[class1_ix]
    class1_y = y[class1_ix]
    kmeans_clusters_class1 = kmeans_clusters[class1_ix]
    em_probs_class1 = em_probs[class1_ix]

    ax1.scatter(
        class0_reduced[:, 0],
        class0_reduced[:, 1],
        c=kmeans_clusters_class0,
        marker="_",
        alpha=0.3,
    )
    ax1.scatter(
        class1_reduced[:, 0],
        class1_reduced[:, 1],
        c=kmeans_clusters_class1,
        marker="+",
        alpha=0.3,
    )
    ax2.scatter(
        class0_reduced[:, 0],
        class0_reduced[:, 1],
        c=em_probs_class0,
        marker="_",
        alpha=0.3,
    )
    ax2.scatter(
        class1_reduced[:, 0],
        class1_reduced[:, 1],
        c=em_probs_class1,
        marker="+",
        alpha=0.3,
    )

    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax1.set_title("K-Means Cluster Projection")
    ax2.set_title("EM Cluster Projection")

    plt.savefig(os.path.join(output_dir, f"{file_prefix}_clusterprojections.png"))
    plt.close()

    return X, kmeans_clusters, em_probs


def plot_cluster_means(
    data_loader,
    output_dir,
    file_prefix,
    kmeans_clusters=2,
    em_clusters=2,
):

    X, y = data_loader()
    if data_loader is load_intention:
        X_plot = X[
            [
                "Administrative",
                "Administrative_Duration",
                "Informational",
                "Informational_Duration",
                "ProductRelated",
                "ProductRelated_Duration",
                "BounceRates",
                "ExitRates",
                "PageValues",
            ]
        ]

    else:
        X_plot = X

    fig, (ax1, ax2) = plt.subplots(1, 2)
    kmeans = KMeans(kmeans_clusters, random_state=1)
    em = GaussianMixture(n_components=em_clusters, random_state=1)
    kmeans.fit(X)
    em.fit(X)

    kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    kmeans_df = kmeans_df[X_plot.columns]
    em_df = pd.DataFrame(em.means_, columns=X.columns)
    em_df = em_df[X_plot.columns]

    kmeans_df.plot(kind="bar", ax=ax1)
    em_df.plot(kind="bar", ax=ax2)

    ax1.set_ylabel("Mean Value")
    ax2.set_ylabel("Mean Value")

    ax1.set_xlabel("Cluster")
    ax2.set_xlabel("Cluster")

    ax1.set_title('K-Means Cluster Centers')
    ax2.set_title('EM Cluster Centers')
    ax1.get_legend().remove()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_clusterprojections.png"))
    plt.close()



if __name__ == "__main__":
    plot_dir = os.path.join("plots", "part1")
    intention_datafile = os.path.join("output", "part1", "exp1_intention_data.json")
    pulsar_datafile = os.path.join("output", "part1", "exp1_pulsar_data.json")
    save_clustering_plots(intention_datafile, "intention")
    save_clustering_plots(pulsar_datafile, "pulsar")
    save_cluster_evaluations_plots(intention_datafile, "intention", plot_dir)
    save_cluster_evaluations_plots(pulsar_datafile, "pulsar", plot_dir)
    plot_cluster_means(load_intention, plot_dir, 'intention')
    plot_cluster_means(load_pulsar, plot_dir, 'pulsar')
    
    print("Intention Evaluation")
    print_evaluation_stats(intention_datafile)

    print("Pulsar Evaluation")
    print_evaluation_stats(pulsar_datafile)
