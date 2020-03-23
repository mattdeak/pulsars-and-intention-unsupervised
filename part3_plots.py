from part1_plots import (
    save_clustering_plots,
    print_clustering_stats,
    print_evaluation_stats,
)
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, FastICA
import os
from utils import load_intention, load_pulsar
from mpl_toolkits.mplot3d import Axes3D
from utils import (
    load_pulsar_PCA_reduced,
    load_intention_PCA_reduced,
    load_pulsar_ICA_reduced,
    load_intention_ICA_reduced,
    load_intention_RP_reduced,
    load_pulsar_RP_reduced,
    load_intention_LLE_reduced,
    load_pulsar_LLE_reduced,
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

data_folder = os.path.join("output", "part3")
plot_dir = os.path.join("plots", "part3")


def get_datafile_path(dataset, reduction_method):
    return os.path.join(data_folder, f"{dataset}_{reduction_method}_clustering.json")


def get_inverse_transform(transformer, X):
    if isinstance(transformer, PCA) or isinstance(transformer, FastICA):
        return transformer.inverse_transform(X)
    elif isinstance(transformer, GaussianRandomProjection):
        pseudo_inverse = np.linalg.pinv(transformer.components_)
        reduced = transformer.transform(X)
        reconstructed = (pseudo_inverse @ reduced.T).T
        return reconstructed
    else:
        raise NotImplementedError("Transformer not supported for inverse transform")


def plot_cluster_means(
    data_loader,
    transformer_path,
    dataset,
    output_dir,
    file_prefix,
    kmeans_clusters=2,
    em_clusters=2,
):
    if dataset == "intention":
        X_untransformed = load_intention()
    else:
        X_untransformed = load_pulsar()

    Xtransformed, y = data_loader()
    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f)

    X = get_inverse_transform(transformer, Xtransformed)
    X = pd.DataFrame(X, columns=X_untransformed.columns)

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

    ax1.set_title("K-Means Cluster Centers")
    ax2.set_title("EM Cluster Centers")
    ax1.get_legend().remove()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_clusterprojections.png"))
    plt.close()


# Pulsar LLE


def main():
    print("Generating Part3 Plots")
    # Intention PCA
    intention_pca, y_intention = load_intention_PCA_reduced()
    em = GaussianMixture(2, random_state=1)

    pulsar_lle, y_pulsar = load_pulsar_LLE_reduced()
    kmeans = KMeans(2, random_state=1)

    em.fit(intention_pca)
    kmeans.fit(pulsar_lle)

    kmeans_clusters = kmeans.predict(pulsar_lle)
    em_probs = em.predict_proba(intention_pca)[:, 0]

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")
    ax1.scatter(
        intention_pca[:, 0],
        intention_pca[:, 1],
        zs=intention_pca[:, 2],
        c=1 - em_probs,
        alpha=0.3,
    )
    ax2.scatter(
        pulsar_lle[:, 0],
        pulsar_lle[:, 1],
        zs=pulsar_lle[:, 2],
        c=1 - kmeans_clusters,
        alpha=0.3,
    )
    ax3.scatter(
        intention_pca[:, 0],
        intention_pca[:, 1],
        zs=intention_pca[:, 2],
        c=y_intention,
        alpha=0.3,
    )
    ax4.scatter(
        pulsar_lle[:, 0], pulsar_lle[:, 1], zs=pulsar_lle[:, 2], c=y_pulsar, alpha=0.3
    )

    ax1.set_xlabel("PCA Dimension 1")
    ax1.set_ylabel("PCA Dimension 2")
    ax1.set_zlabel("PCA Dimension 3")
    ax2.set_xlabel("LLE Dimension 1")
    ax2.set_ylabel("LLE Dimension 2")
    ax2.set_zlabel("LLE Dimension 3")
    ax3.set_xlabel("PCA Dimension 1")
    ax3.set_ylabel("PCA Dimension 2")
    ax3.set_zlabel("PCA Dimension 3")
    ax4.set_xlabel("LLE Dimension 1")
    ax4.set_ylabel("LLE Dimension 2")
    ax4.set_zlabel("LLE Dimension 3")

    ax1.set_title("EM-Predicted Clusters on PCA")
    ax2.set_title("K-Means Predicted Clusters on LLE")

    ax3.set_title("True Labels in PCA Embedding")
    ax4.set_title("True Labels in LLE Embedding")

    plot_dir = os.path.join("plots", "part3")
    plt.savefig(os.path.join(plot_dir, "BestClustering.png"))
    plt.close()

    intention_pca_datafile = get_datafile_path("intention", "pca")
    intention_ica_datafile = get_datafile_path("intention", "ica")
    intention_rp_datafile = get_datafile_path("intention", "rp")
    intention_lle_datafile = get_datafile_path("intention", "lle")
    pulsar_pca_datafile = get_datafile_path("pulsar", "pca")
    pulsar_ica_datafile = get_datafile_path("pulsar", "ica")
    pulsar_rp_datafile = get_datafile_path("pulsar", "rp")
    pulsar_lle_datafile = get_datafile_path("pulsar", "lle")

    save_clustering_plots(intention_pca_datafile, "intention_pca", plot_dir)
    save_clustering_plots(intention_ica_datafile, "intention_ica", plot_dir)
    save_clustering_plots(intention_rp_datafile, "intention_rp", plot_dir)
    save_clustering_plots(intention_lle_datafile, "intention_lle", plot_dir)

    save_clustering_plots(pulsar_pca_datafile, "pulsar_pca", plot_dir)
    save_clustering_plots(pulsar_ica_datafile, "pulsar_ica", plot_dir)
    save_clustering_plots(pulsar_rp_datafile, "pulsar_rp", plot_dir)
    save_clustering_plots(pulsar_lle_datafile, "pulsar_lle", plot_dir)

    print("Intention pca results")
    print_evaluation_stats(intention_pca_datafile)
    print()

    print("Intention ica results")
    print_evaluation_stats(intention_ica_datafile, kmeans_clusters=3)
    print()

    print("Intention rp results")
    print_evaluation_stats(intention_rp_datafile)
    print()

    print("Intention lle results")
    print_evaluation_stats(intention_lle_datafile)
    print()

    print("pulsar pca results")
    print_evaluation_stats(pulsar_pca_datafile)
    print()

    print("pulsar ica results")
    print_evaluation_stats(pulsar_ica_datafile)
    print()

    print("pulsar rp results")
    print_evaluation_stats(pulsar_rp_datafile)
    print()

    print("pulsar lle results")
    print_evaluation_stats(pulsar_lle_datafile)
    print()

    pulsar_PCA_X, pulsar_PCA_y = load_pulsar_PCA_reduced()
    intention_PCA_X, intention_PCA_y = load_intention_PCA_reduced()

    pulsar_ICA_X, pulsar_ICA_y = load_pulsar_ICA_reduced()
    intention_ICA_X, intention_ICA_y = load_intention_ICA_reduced()

    pulsar_RP_X, pulsar_RP_y = load_pulsar_RP_reduced()
    intention_RP_X, intention_RP_y = load_intention_RP_reduced()

    pulsar_LLE_X, pulsar_LLE_y = load_pulsar_LLE_reduced()
    intention_LLE_X, intention_LLE_y = load_intention_LLE_reduced()

    datafile = os.path.join(data_folder, "intention_pca_clustering.json")

    print_clustering_stats(intention_LLE_X, intention_LLE_y)


if __name__ == "__main__":
    main()
