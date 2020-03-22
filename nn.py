from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from utils import (
    load_intention,
    load_intention_PCA_reduced,
    load_intention_ICA_reduced,
    load_intention_RP_reduced,
    load_intention_LLE_reduced,
)
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

np.random.seed(1)


def load_part4_network(transformer):
    network_path = os.path.join("output", "part4", f"{transformer}_tuned_network.pkl")
    assert os.path.exists(
        network_path
    ), "Make sure part4 experiments have been run before using this function"
    with open(network_path, "rb") as f:
        clf = pickle.load(f)

    return clf


def tune_part4networks():
    neural_params = {
        "hidden_layer_sizes": [(64, 64), (128, 128), (64, 64, 64), (128, 128, 128)],
        "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    }

    for transformer, data_func in zip(
        ["None", "PCA", "ICA", "RP", "LLE"],
        [
            load_intention,
            load_intention_PCA_reduced,
            load_intention_ICA_reduced,
            load_intention_RP_reduced,
            load_intention_LLE_reduced,
        ],
    ):
        print(f"Tuning network on {transformer} reduced data")
        X, y = data_func()

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=0.8, random_state=1
        )

        clf = GridSearchCV(
            MLPClassifier(max_iter=2000, early_stopping=True),
            neural_params,
            scoring="f1",
        )
        clf.fit(Xtrain, ytrain)

        model_savepath = os.path.join(
            "output", "part4", f"{transformer}_tuned_network.pkl"
        )

        with open(model_savepath, "wb") as f:
            pickle.dump(clf, f)

def load_part5_network(variant='minimal'):
    network_path = os.path.join("output", "part5", f"clf_{variant}.pkl")
    assert os.path.exists(
        network_path
    ), "Make sure part4 experiments have been run before using this function"
    with open(network_path, "rb") as f:
        clf = pickle.load(f)

    return clf

def generate_cluster_dfs():
    X, y = load_intention()

    kmeans = KMeans(2)
    em = GaussianMixture(2)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    kmeans.fit(Xtrain)
    em.fit(Xtrain)

    cluster1, cluster2 = kmeans.cluster_centers_

    distance_from_cluster1 = (np.sqrt((X - cluster1) ** 2)).mean(axis=1)
    distance_from_cluster2 = (np.sqrt((X - cluster2) ** 2)).mean(axis=1)
    point_probabilities = em.predict_proba(X)

    X_minimal = pd.DataFrame(
        {
            "dist1": distance_from_cluster1,
            "dist2": distance_from_cluster2,
            "prob1": point_probabilities[:, 0],
            "prob2": point_probabilities[:, 1],
        }
    )
    X_augmented = X.copy()
    X_augmented[["dist1", "dist2", "prob1", "prob2"]] = X_minimal.copy()
    return X_minimal, X_augmented, y

def evaluate_part5networks():
    variants = ['minimal','augmented']
    X_minimal, X_augmented, y = generate_cluster_dfs()
    results = {}
    for variant, X in zip(variants, [X_minimal, X_augmented]):
        clf = load_part5_network(variant=variant)
        # Should be consistent due to random seed

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=0.8, random_state=1
        )

        train_preds = clf.predict(Xtrain)
        test_preds = clf.predict(Xtest)

        train_acc = accuracy_score(ytrain, train_preds)
        test_acc = accuracy_score(ytest, test_preds)

        train_f1 = f1_score(ytrain, train_preds)
        test_f1 = f1_score(ytest, test_preds)

        best_classifier_index = np.argmin(clf.cv_results_["rank_test_score"])

        train_time = clf.cv_results_["mean_fit_time"][best_classifier_index]

        results[variant] = {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "train_time": train_time,
        }

    return results

def evaluate_part4network(transformer):
    clf = load_part4_network(transformer)
    if transformer == "None":
        X, y = load_intention()
    elif transformer == "PCA":
        X, y = load_intention_PCA_reduced()
    elif transformer == "ICA":
        X, y = load_intention_ICA_reduced()
    elif transformer == "RP":
        X, y = load_intention_RP_reduced()
    elif transformer == "LLE":
        X, y = load_intention_LLE_reduced()

    # Should be consistent due to random seed
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    train_preds = clf.predict(Xtrain)
    test_preds = clf.predict(Xtest)

    train_acc = accuracy_score(ytrain, train_preds)
    test_acc = accuracy_score(ytest, test_preds)

    train_f1 = f1_score(ytrain, train_preds)
    test_f1 = f1_score(ytest, test_preds)

    best_classifier_index = np.argmin(clf.cv_results_["rank_test_score"])

    train_time = clf.cv_results_["mean_fit_time"][best_classifier_index]

    return {
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
    }


def tune_part5networks():
    # TODO
    neural_params = {
        "hidden_layer_sizes": [(64, 64), (128, 128), (64, 64, 64), (128, 128, 128)],
        "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    }
    X, y = load_intention()

    kmeans = KMeans(2)
    em = GaussianMixture(2)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    kmeans.fit(Xtrain)
    em.fit(Xtrain)

    cluster1, cluster2 = kmeans.cluster_centers_

    distance_from_cluster1 = (np.sqrt((X - cluster1) ** 2)).mean(axis=1)
    distance_from_cluster2 = (np.sqrt((X - cluster2) ** 2)).mean(axis=1)
    point_probabilities = em.predict_proba(X)

    X_minimal = pd.DataFrame(
        {
            "dist1": distance_from_cluster1,
            "dist2": distance_from_cluster2,
            "prob1": point_probabilities[:, 0],
            "prob2": point_probabilities[:, 1],
        }
    )
    X_augmented = X.copy()
    X_augmented[["dist1", "dist2", "prob1", "prob2"]] = X_minimal.copy()

    Xminimaltrain, Xminimaltest = train_test_split(
        X_minimal, train_size=0.8, random_state=1
    )
    Xaugtrain, Xaugtest = train_test_split(X_augmented, train_size=0.8, random_state=1)

    mlp = MLPClassifier(max_iter=2000, early_stopping=True)

    print("Tuning neural network on minimal dataset")
    clf_minimal = GridSearchCV(mlp, neural_params)
    clf_minimal.fit(X_minimal, y)

    print("Tuning neural network on augmented dataset")
    clf_augmented = GridSearchCV(
        MLPClassifier(max_iter=2000, early_stopping=True), neural_params
    )
    clf_augmented.fit(X_augmented, y)

    minimal_outpath = os.path.join("output", "part5", "clf_minimal.pkl")
    augmented_outpath = os.path.join("output", "part5", "clf_augmented.pkl")

    with open(minimal_outpath, "wb") as f:
        pickle.dump(clf_minimal, f)

    with open(augmented_outpath, "wb") as f:
        pickle.dump(clf_augmented, f)


def print_best_params(tranformers=["None", "PCA", "ICA", "RP", "LLE"]):
    for transformer in tranformers:
        clf = load_part4_network(transformer)
        print(f"Best Params with transformer {transformer}: {clf.best_params_}")

def print_best_params_part5(variants=['minimal','augmented']):
    for variant in variants:
        clf = load_part5_network(variant)
        print(f"Best Params with variant {variant}: {clf.best_params_}")



def get_all_evaluations(transformers=["None", "PCA", "ICA", "RP", "LLE"]):
    results = {}

    for transformer in transformers:
        results[transformer] = evaluate_part4network(transformer)

    results_df = pd.DataFrame.from_dict(results)
    return results_df


if __name__ == "__main__":
    pass
    # tune_networks()
    # print_best_params()
