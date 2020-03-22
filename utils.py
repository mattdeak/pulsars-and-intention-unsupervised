import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def load_intention():
    train = pd.read_csv("data/online_shoppers_intention.csv")
    train.drop_duplicates(inplace=True)
    train.dropna(inplace=True)

    train.OperatingSystems = train.OperatingSystems.astype("category")
    train.Browser = train.Browser.astype("category")
    train.Region = train.Region.astype("category")
    train.TrafficType = train.TrafficType.astype("category")

    # Normalize numeric columns
    numeric_cols = train.select_dtypes(include=["float64"]).columns.tolist()
    train[numeric_cols] = train[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

    # numerical_features = [col for col in train.columns if train.get
    train["Weekend"] = train["Weekend"].astype("category")
    train = pd.get_dummies(train, drop_first=True)
    target = train.Revenue.astype("int8")

    train.drop("Revenue", axis=1, inplace=True)
    return train, target


def load_pulsar():
    train = pd.read_csv("data/pulsar_stars.csv")
    target = train.target_class
    train.drop("target_class", axis=1, inplace=True)
    train = (train - train.mean()) / train.std()
    return train, target


def load_pulsar_PCA_reduced(keep_pct=0.9):
    filepath = os.path.join("output", "part2", "PCA", "pulsar_PCA.npy")
    pca_object_path = os.path.join(
        "output", "part2", "PCA", "pulsar_PCA_transformer.pkl"
    )
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)

    with open(pca_object_path, 'rb') as f:
        pca = pickle.load(f)

    explained_variance_ratios = pca.explained_variance_ratio_
    trimmed_data = trim_PCA_reduction(data, explained_variance_ratios, info_keep=keep_pct)
    labels = get_labels("pulsar")
    return trimmed_data, labels


def load_intention_PCA_reduced(keep_pct=0.9):
    filepath = os.path.join("output", "part2", "PCA", "intention_PCA.npy")
    pca_object_path = os.path.join(
        "output", "part2", "PCA", "intention_PCA_transformer.pkl"
    )
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)

    with open(pca_object_path, 'rb') as f:
        pca = pickle.load(f)

    explained_variance_ratios = pca.explained_variance_ratio_
    trimmed_data = trim_PCA_reduction(data, explained_variance_ratios, info_keep=keep_pct)
    labels = get_labels("intention")
    return trimmed_data, labels


def load_pulsar_ICA_reduced(components=4):
    filepath = os.path.join("output", "part2", "ICA", f"pulsar_ICA{components}.npy")
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)
    labels = get_labels("pulsar")
    return data, labels


def load_intention_ICA_reduced(components=9):
    filepath = os.path.join("output", "part2", "ICA", f"intention_ICA{components}.npy")
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)
    labels = get_labels("intention")
    return data, labels


def load_pulsar_RP_reduced(n_components=3):
    filepath = os.path.join("output", "part2", "RP", f"pulsar_RP{n_components}_trial1.npy")
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)
    labels = get_labels("pulsar")
    return data, labels


def load_intention_RP_reduced(n_components=9):
    filepath = os.path.join("output", "part2", "RP", f"intention_RP{n_components}_trial1.npy")
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)
    labels = get_labels("intention")
    return data, labels


def load_pulsar_LLE_reduced():
    filepath = os.path.join("output", "part2", "LLE", "pulsar_LLE3.npy")
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)

    labels = get_labels("pulsar")
    return data, labels


def load_intention_LLE_reduced():
    filepath = os.path.join("output", "part2", "LLE", "intention_LLE6.npy")
    assert os.path.exists(
        filepath
    ), "Must run part 2 experiments before calling this function"

    with open(filepath, "rb") as f:
        data = np.load(filepath)

    labels = get_labels("intention")
    return data, labels


def get_labels(dataset):
    if dataset == "pulsar":
        _, y = load_pulsar()
    else:
        _, y = load_intention()

    return y


def get_true_data(dataset):
    if dataset == "pulsar":
        X, _ = load_pulsar()
    else:
        X, _ = load_intention()
    return X


def trim_PCA_reduction(reduced, explained_variance_ratios, info_keep=0.9, dims=None):
    assert (
        info_keep is not None or dims is not None
    ), "Must specify either info_keep or dims"
    assert not (
        info_keep is not None and dims is not None
    ), "Can't use both info_keep or dims"

    cumvar = np.cumsum(explained_variance_ratios)
    pca_dims = len(cumvar[cumvar < info_keep])
    trimmed = reduced[:, :pca_dims]

    return trimmed
