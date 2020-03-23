import os
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import procrustes
import pandas as pd
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from utils import load_intention, load_pulsar, get_true_data, get_labels
from scipy.stats import kurtosis

part2_plot_dir = os.path.join("plots", "part2")

data_dir = os.path.join("output", "part2")
ica_dir = os.path.join(data_dir, "ICA")
pca_dir = os.path.join(data_dir, "PCA")
rp_dir = os.path.join(data_dir, "RP")
lle_dir = os.path.join(data_dir, "LLE")

# dataset = "intention"
dataset = "pulsar"
#
ica_components_map = {"pulsar": 4, "intention": 9}  # From analysis
rp_components_map = {"intention": 20}
lle_components_map = {"intention": 6, "pulsar": 3}


def plot_ICA_components(dataset):
    if dataset == "intention":
        X, y = load_intention()
        cols_to_use = [
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
    else:
        X, y = load_pulsar()
        cols_to_use = X.columns

    n_components = ica_components_map[dataset]
    transformer_name = f"{dataset}_ICA{n_components}_transformer.pkl"
    folder = os.path.join("output", "part2", "ICA")

    transformer = get_transformer(transformer_name, folder)
    components = pd.DataFrame(transformer.components_, columns=X.columns)
    components = components[cols_to_use]
    rows = int(np.sqrt(n_components))

    fig = plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size": 5})
    for i in range(n_components):
        ax = fig.add_subplot(int(f"{rows}{rows}{i+1}"))
        ax.set_title(f"Component {i+1}")
        components.loc[i, :].plot(kind="bar", ax=ax)

        # Turn off ticks
        #
        if n_components - i > rows:
            ax.set_xticklabels([])
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
    plt.subplots_adjust(wspace=1)
    plt.rcParams.update({"font.size": 14})
    fig.suptitle("ICA Components")

    plot_dir = os.path.join("plots", "part2", f"ica_components{dataset}.png")
    plt.savefig(plot_dir)
    plt.close()


def get_transformer(filename, folder):
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_transformed_data(dataset_name, transformer_name, folder):
    filepath = os.path.join(folder, f"{dataset_name}_{transformer_name}.npy")
    with open(filepath, "rb") as f:
        return np.load(f)


def plot_pca_points(dataset):
    if dataset == "pulsar":
        _, y = load_pulsar()
    else:
        _, y = load_intention()

    pca_data = get_transformed_data(dataset, "PCA", pca_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["red", "blue"]
    cmap = mpl.colors.ListedColormap(colors)
    ax.scatter(
        pca_data[:, 0], pca_data[:, 1], zs=pca_data[:, 2], c=y, cmap=cmap, alpha=0.2
    )
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    ax.set_zlabel("PCA Dimension 3")

    title = "PCA-Transformed Data Points by Label"
    ax.set_title(title)
    plt.savefig(os.path.join(part2_plot_dir, f"PCAPlot{dataset}.png"))
    plt.close()


def plot_pca_explained_variance(dataset):
    pca_transformer_name = f"{dataset}_PCA_transformer.pkl"
    transformer = get_transformer(pca_transformer_name, pca_dir)

    explained_variance_ratio_ = transformer.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(transformer.explained_variance_ratio_)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    xs = np.arange(len(explained_variance_ratio_))
    width = 0.4

    ax.bar(
        xs - width / 2,
        explained_variance_ratio_,
        width,
        color="green",
        label="Explained Variance",
    )
    ax.bar(
        xs + width / 2,
        cumulative_explained_variance,
        width,
        label="Cumulative Explained Variance",
    )

    ax.set_title("Variance in Data Explained by PCA Components")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Explained Variance")
    ax.set_xlim((-0.5, xs[-1]))
    ax.set_ylim((0, 1))

    ax.legend()
    plt.savefig(os.path.join(part2_plot_dir, f"PCAExplainedVariance{dataset}.png"))
    plt.close()


def plot_principal_axes(dataset):
    if dataset == "pulsar":
        X, y = load_pulsar()
        cols_to_use = X.columns  # Use all
    else:
        X, y = load_intention()
        cols_to_use = [
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

    index = [f"Component {i+1}" for i in range(X.shape[1])]
    pca_transformer_name = f"{dataset}_PCA_transformer.pkl"
    pca_data = get_transformed_data(dataset, "PCA", pca_dir)
    pca = get_transformer(pca_transformer_name, pca_dir)
    components = pd.DataFrame(pca.components_, columns=X.columns, index=index)

    components = components[cols_to_use]
    first_two_components = components.iloc[:2]
    ax = first_two_components.plot(kind="bar")
    ax.set_ylabel("Value")

    ax.set_title("Principal Axes in Feature Space resulting from PCA Decomposition")
    plt.savefig(os.path.join(part2_plot_dir, f"PrincipalAxes{dataset}.png"))
    plt.close()


def extract_number(string, transformer):
    """Extracts the number present in a filename according to scheme laid out
    This was done as an afterthought and so is stupid. Please forgive me.

    Parameters
    ----------

    string : filename
    transformer : name of transformer

    Returns
    -------
    """
    n = len(transformer)
    transformer_location = string.find(transformer)

    period_location = string.find(".", transformer_location)
    underscore_location = string.find("_", transformer_location)

    if period_location == -1:
        end_location = underscore_location
    elif underscore_location == -1:
        end_location = period_location
    else:
        end_location = min(underscore_location, period_location)

    return int(string[transformer_location + n : end_location])


def get_kurtoses(dataset, ica_folder="output/part2/ICA"):
    ica_transformers = [
        t for t in os.listdir(ica_folder) if "pkl" in t and dataset in t
    ]
    ica_data = [
        t
        for t in os.listdir(ica_folder)
        if "npy" in t and dataset in t and "tsne" not in t
    ]

    ica_transformers = sorted(ica_transformers, key=lambda x: extract_number(x, "ICA"))
    ica_data = sorted(ica_data, key=lambda x: extract_number(x, "ICA"))
    numbers = [extract_number(t, "ICA") for t in ica_data]

    kurtoses = []
    for t_name in ica_transformers:
        t = get_transformer(t_name, "output/part2/ICA")
        ica_kurtosis = kurtosis(t.components_, axis=1, fisher=False)
        avg_kurtosis = np.mean(ica_kurtosis)
        kurtoses.append(avg_kurtosis)

    return numbers, kurtoses


def plot_ica_kurtoses(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    components, kurtoses = get_kurtoses(dataset)

    ax.plot(components, kurtoses, marker="o")
    ax.set_ylabel("Average Kurtosis of Components")
    ax.set_xlabel("Number of Components")
    ax.set_title("Average Kurtosis Across Number of Components")
    ax.xaxis.set_major_locator(MultipleLocator())
    ax.grid()
    ax.set_xlim((components[0], components[-1]))
    plt.savefig(os.path.join(part2_plot_dir, f"AverageKurtosisICA{dataset}.png"))
    plt.close()


def generate_tsnes(dataset, transformer="ICA"):
    print(f"Generating TSNE data for {dataset} on transformer {transformer}")
    if transformer == "ICA":
        components = ica_components_map[dataset]
        data_filepath = os.path.join(ica_dir, f"{dataset}_ICA{components}.npy")
    elif transformer == "LLE":
        components = lle_components_map[dataset]
        data_filepath = os.path.join(lle_dir, f"{dataset}_LLE{components}.npy")
    else:
        raise NotImplemented("Transformer not implemented. Must be ICA or LLE")

    raw_filepath = data_filepath.replace(".npy", "")
    tsne2 = TSNE(n_components=2)
    tsne3 = TSNE(n_components=3)
    with open(data_filepath, "rb") as f:
        data = np.load(f)

    transformed2 = tsne2.fit_transform(data)
    transformed3 = tsne3.fit_transform(data)

    output2 = raw_filepath + "_tsne2.npy"
    output3 = raw_filepath + "_tsne3.npy"

    with open(output2, "wb") as f:
        np.save(f, transformed2)

    with open(output3, "wb") as f:
        np.save(f, transformed3)


def plot_ica_tsnes(dataset):
    data_dir = "output/part2/ICA"
    ica_components = ica_components_map[dataset]
    filepath = os.path.join(data_dir, f"{dataset}_ICA{ica_components}.npy")

    tsne3_path = filepath.replace(".npy", "") + "_tsne3.npy"
    tsne2_path = filepath.replace(".npy", "") + "_tsne2.npy"

    with open(tsne3_path, "rb") as f:
        tsne3_data = np.load(f)

    with open(tsne2_path, "rb") as f:
        tsne2_data = np.load(f)

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection="3d")

    labels = get_labels(dataset)
    ax1.scatter(tsne2_data[:, 0], tsne2_data[:, 1], c=labels, alpha=0.3)

    ax2.scatter(
        tsne3_data[:, 0], tsne3_data[:, 1], zs=tsne3_data[:, 2], c=labels, alpha=0.3
    )

    ax1.set_title("2-D TSNE Embedding of ICA Transform (9 Components)")
    ax2.set_title("3-D TSNE Embedding of ICA Transform (9 Components)")

    ax1.set_xlabel("TSNE Dimension 1")
    ax1.set_ylabel("TSNE Dimension 2")
    ax2.set_xlabel("TSNE Dimension 1")
    ax2.set_ylabel("TSNE Dimension 2")
    ax2.set_zlabel("TSNE Dimension 3")
    plt.savefig(os.path.join(part2_plot_dir, f"TSNE_ICA{dataset}.png"))
    plt.close()


def plot_lle_tsnes(dataset):
    components = lle_components_map[dataset]
    filepath = os.path.join(lle_dir, f"{dataset}_LLE{components}.npy")

    tsne3_path = filepath.replace(".npy", "") + "_tsne3.npy"
    tsne2_path = filepath.replace(".npy", "") + "_tsne2.npy"

    with open(tsne3_path, "rb") as f:
        tsne3_data = np.load(f)

    with open(tsne2_path, "rb") as f:
        tsne2_data = np.load(f)

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection="3d")

    labels = get_labels(dataset)
    ax1.scatter(tsne2_data[:, 0], tsne2_data[:, 1], c=labels, alpha=0.3)

    ax2.scatter(
        tsne3_data[:, 0], tsne3_data[:, 1], zs=tsne3_data[:, 2], c=labels, alpha=0.3
    )

    ax1.set_title("2-D TSNE Embedding of ICA Transform (9 Components)")
    ax2.set_title("3-D TSNE Embedding of ICA Transform (9 Components)")

    ax1.set_xlabel("TSNE Dimension 1")
    ax1.set_ylabel("TSNE Dimension 2")
    ax2.set_xlabel("TSNE Dimension 1")
    ax2.set_ylabel("TSNE Dimension 2")
    ax2.set_zlabel("TSNE Dimension 3")
    plt.savefig(os.path.join(part2_plot_dir, f"TSNE_LLE{dataset}.png"))
    plt.close()


def get_RP_reconstruction_errors(dataset):
    X = get_true_data(dataset)
    trials = [f for f in os.listdir(rp_dir) if dataset in f]
    projections_by_components = [extract_number(x, "RP") for x in trials]
    unique_projections_number = np.unique(projections_by_components)

    results = defaultdict(list)
    for number in tqdm(unique_projections_number):
        proj_trials = [
            t
            for t in trials
            if extract_number(t, "RP") == number and "transformer" in t
        ]

        for trial in proj_trials:
            transformer_path = os.path.join(rp_dir, trial)
            with open(transformer_path, "rb") as f:
                transformer = pickle.load(f)

            pseudo_inverse = np.linalg.pinv(transformer.components_)
            reduced = transformer.transform(X)
            reconstructed = (pseudo_inverse @ reduced.T).T
            results[number].append(((X - reconstructed) ** 2).mean().mean())
    return results


def plot_reconstruction_error(dataset):
    print("Collecting Reconstruction Errors")
    reconstruction_errors = get_RP_reconstruction_errors(dataset)

    xs = list(reconstruction_errors.keys())
    std_reconstruction_errors = np.array(
        [np.std(v) for v in reconstruction_errors.values()]
    )
    mean_reconstruction_errors = np.array(
        [np.mean(v) for v in reconstruction_errors.values()]
    )

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(xs, mean_reconstruction_errors, marker="o")
    ax.fill_between(
        xs,
        mean_reconstruction_errors - std_reconstruction_errors,
        mean_reconstruction_errors + std_reconstruction_errors,
        alpha=0.3,
    )

    ax.set_xlim((xs[0], xs[-1]))
    ax.set_title("Average Reconstruction Error for RP across # of Components")
    ax.set_xlabel("# of Components")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator())
    plt.savefig(os.path.join(part2_plot_dir, f"RPReconstructionErrors{dataset}.png"))
    plt.close()


def get_LLE_procrustes(dataset):
    print("Collecting Procrustes from LLE Embeddings")
    X = get_true_data(dataset)
    trials = [f for f in os.listdir(lle_dir) if dataset in f and "pkl" in f]
    trials = sorted(trials, key=lambda x: extract_number(x, "LLE"))

    results = {}
    for trial in tqdm(trials):
        transformer_path = os.path.join(lle_dir, trial)
        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)

        reduced = transformer.transform(X)
        # Pad reduced with zeros to match dimensionality of X
        dims = reduced.shape[1]
        padded_reduced = np.zeros_like(X)
        padded_reduced[:, : reduced.shape[1]] = reduced

        _, _, disparity = procrustes(X, padded_reduced)
        results[dims] = disparity

    return results


def plot_procrustes_disparities(dataset):
    print("Collecting Reconstruction Errors")
    procrustes_dict = get_LLE_procrustes(dataset)

    xs = list(procrustes_dict.keys())
    disparities = list(procrustes_dict.values())

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(xs, disparities, marker="o")

    ax.set_xlim((xs[0], xs[-1]))
    ax.set_title("Procrustes Disparity for LLE Embedding of N-Dimensions")
    ax.set_xlabel("# of Dimensions")
    ax.set_ylabel("Procrustes Disparity")
    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator())
    plt.savefig(os.path.join(part2_plot_dir, f"LLEProcrustesDisparity{dataset}.png"))
    plt.close()


if __name__ == "__main__":
    print('Generating Part 2 Plots')
    for dataset in ['pulsar','intention']:
        plot_pca_points(dataset)
        plot_pca_explained_variance(dataset)
        plot_principal_axes(dataset)
        plot_ica_kurtoses(dataset)
        plot_ica_tsnes(dataset)
        plot_reconstruction_error(dataset)
        plot_procrustes_disparities(dataset)
        plot_lle_tsnes(dataset)
