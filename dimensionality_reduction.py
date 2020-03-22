from sklearn.decomposition import PCA, FastICA, DictionaryLearning
from tqdm import tqdm
import os
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
from utils import load_intention, load_pulsar
import pickle

RANDOM_STATE = 1

def save_transformed_data(data, transformer, outpath):
    transformed = transformer.fit_transform(data)
    with open(outpath, "wb") as f:
        np.save(f, transformed)


def run_PCA_collection(data, file_prefix, output_dir):
    pca = PCA(svd_solver="full", random_state=RANDOM_STATE)
    outpath = os.path.join(output_dir, f"{file_prefix}_PCA.npy")
    save_transformed_data(data, pca, outpath)
    transformer_filepath = os.path.join(
        output_dir, f"{file_prefix}_PCA_transformer.pkl"
    )
    with open(transformer_filepath, "wb") as f:
        pickle.dump(pca, f)


def run_ICA_collection(data, file_prefix, output_dir, max_component_ratio=0.3):
    max_components = int(data.shape[1] * max_component_ratio)
    for i in tqdm(range(2, max_components + 1)):
        outpath = os.path.join(output_dir, f"{file_prefix}_ICA{i}.npy")
        ica = FastICA(n_components=i, max_iter=1000, random_state=RANDOM_STATE)
        save_transformed_data(data, ica, outpath)
        transformer_filepath = os.path.join(
            output_dir, f"{file_prefix}_ICA{i}_transformer.pkl"
        )
        with open(transformer_filepath, "wb") as f:
            pickle.dump(ica, f)


def run_RP_collection(
    data, file_prefix, output_dir, max_trials=30, max_component_ratio=0.3
):
    components = int(data.shape[1] * max_component_ratio)
    for i in tqdm(range(2, components+1)):
        for j in range(max_trials):
            outpath = os.path.join(output_dir, f"{file_prefix}_RP{i}_trial{j+1}.npy")
            rp = GaussianRandomProjection(n_components=i, random_state=j)
            save_transformed_data(data, rp, outpath)
            transformer_filepath = os.path.join(
                output_dir, f"{file_prefix}_RP{i}_trial{j+1}_transformer.pkl"
            )
            with open(transformer_filepath, "wb") as f:
                pickle.dump(rp, f)


def run_LLE_collection(data, file_prefix, output_dir, max_component_ratio=0.3):
    max_components = int(data.shape[1] * max_component_ratio)
    for i in tqdm(range(2, max_components + 1)):
        lle = LocallyLinearEmbedding(n_components=i, random_state=RANDOM_STATE)
        outpath = os.path.join(output_dir, f"{file_prefix}_LLE{i}.npy")
        save_transformed_data(data, lle, outpath)
        transformer_filepath = os.path.join(output_dir, f"{file_prefix}_LLE{i}.pkl")
        with open(transformer_filepath, "wb") as f:
            pickle.dump(lle, f)


def save_all_transformations(data, file_prefix, output_dir="output/part2", ica_max_component_ratio=0.3, rp_component_ratio=0.3, lle_max_component_ratio=0.3):
    pca_dir = os.path.join(output_dir, "PCA")
    ica_dir = os.path.join(output_dir, "ICA")
    rp_dir = os.path.join(output_dir, "RP")
    lle_dir = os.path.join(output_dir, "LLE")

    print(f"Running PCA Collection on {file_prefix}")
    # run_PCA_collection(data, file_prefix, pca_dir)
    print(f"Running ICA Collection on {file_prefix}")
    # run_ICA_collection(data, file_prefix, ica_dir, max_component_ratio=ica_max_component_ratio)
    print(f"Running RP Collection on {file_prefix}")
    run_RP_collection(data, file_prefix, rp_dir, max_component_ratio=rp_component_ratio)
    print(f"Running LLE Collection on {file_prefix}")
    # run_LLE_collection(data, file_prefix, lle_dir, max_component_ratio=lle_max_component_ratio)




if __name__ == "__main__":
    X, y = load_intention()
    X2, y2 = load_pulsar()
    print("Collecting Exp2 on Intention")
    save_all_transformations(X, "intention")

    print("Collecting Exp2 Data on Pulsar")
    save_all_transformations(X2, "pulsar")
