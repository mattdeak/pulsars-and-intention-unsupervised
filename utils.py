import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler



def load_intention():
    train = pd.read_csv("data/online_shoppers_intention.csv")
    train.drop_duplicates(inplace=True)
    train.dropna(inplace=True)

    train.OperatingSystems = train.OperatingSystems.astype('category')
    train.Browser = train.Browser.astype('category')
    train.Region = train.Region.astype('category')
    train.TrafficType = train.TrafficType.astype('category')

    # Normalize numeric columns
    numeric_cols = train.select_dtypes(include=['float64']).columns.tolist()
    train[numeric_cols] = train[numeric_cols].apply(lambda x: (x - x.mean())/x.std())
    

    # numerical_features = [col for col in train.columns if train.get
    train['Weekend'] = train['Weekend'].astype('category')
    train = pd.get_dummies(train, drop_first=True)
    target = train.Revenue.astype('int8')



    train.drop("Revenue", axis=1, inplace=True)
    return train, target

def load_pulsar():
    train = pd.read_csv('data/pulsar_stars.csv')
    target = train.target_class
    train.drop('target_class', axis=1, inplace=True)
    train = (train - train.mean())/train.std()
    return train, target

def load_pulsar_PCA_reduced(keep_pct=0.9):
    filepath = os.path.join('output','part2','PCA','pulsar_PCA.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)

    labels = get_labels('pulsar')
    return data, labels


def load_intention_PCA_reduced(keep_pct=0.9):
    filepath = os.path.join('output','part2','PCA','intention_PCA.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)

    labels = get_labels('intention')
    return data, labels


def load_pulsar_ICA_reduced(components=4):
    filepath = os.path.join('output','part2','ICA',f'pulsar_ICA{components}.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)
    labels = get_labels('pulsar')
    return data, labels

def load_intention_ICA_reducted(components=9):
    filepath = os.path.join('output','part2','ICA',f'intention_ICA{components}.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)
    labels = get_labels('intention')
    return data, labels

def load_pulsar_RP_reduced():
    filepath = os.path.join('output','part2','RP','pulsar_PCA.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)
    labels = get_labels('pulsar')
    return data, labels

def load_intention_RP_reduced():
    filepath = os.path.join('output','part2','RP','pulsar_PCA.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)
    labels = get_labels('intention')
    return data, labels

def load_pulsar_LLE_reduced():
    filepath = os.path.join('output','part2','LLE','pulsar_LLE3.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    labels = get_labels('pulsar')
    return data, labels
    with open(filepath, 'rb') as f:
        data = np.load(filepath)

    labels = get_labels('pulsar')
    return data, labels

def load_intention_LLE_reduced():
    filepath = os.path.join('output','part2','PCA','intention_LLE6.npy')
    assert os.path.exists(filepath), "Must run part 2 experiments before calling this function"

    with open(filepath, 'rb') as f:
        data = np.load(filepath)

    labels = get_labels('intention')
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
