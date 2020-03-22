from part1_plots import save_clustering_plots, print_clustering_stats, print_evaluation_stats
import os
from utils import (
    load_pulsar_PCA_reduced,
    load_intention_PCA_reduced,
    load_pulsar_ICA_reduced,
    load_intention_ICA_reduced,
    load_intention_RP_reduced,
    load_pulsar_RP_reduced,
    load_intention_LLE_reduced,
    load_pulsar_LLE_reduced
)

data_folder = os.path.join("output", "part3")
plot_dir = os.path.join('plots','part3')

def get_datafile_path(dataset, reduction_method):
    return os.path.join(data_folder, f'{dataset}_{reduction_method}_clustering.json')

# Clustering Results
intention_pca_datafile = get_datafile_path('intention','pca')
intention_ica_datafile = get_datafile_path('intention','ica')
intention_rp_datafile = get_datafile_path('intention','rp')
intention_lle_datafile = get_datafile_path('intention','lle')
pulsar_pca_datafile = get_datafile_path('pulsar','pca')
pulsar_ica_datafile = get_datafile_path('pulsar','ica')
pulsar_rp_datafile = get_datafile_path('pulsar','rp')
pulsar_lle_datafile = get_datafile_path('pulsar','lle')

save_clustering_plots(intention_pca_datafile, 'intention_pca', plot_dir)
save_clustering_plots(intention_ica_datafile, 'intention_ica', plot_dir)
save_clustering_plots(intention_rp_datafile, 'intention_rp', plot_dir)
save_clustering_plots(intention_lle_datafile, 'intention_lle', plot_dir)

save_clustering_plots(pulsar_pca_datafile, 'pulsar_pca', plot_dir)
save_clustering_plots(pulsar_ica_datafile, 'pulsar_ica', plot_dir)
save_clustering_plots(pulsar_rp_datafile, 'pulsar_rp', plot_dir)
save_clustering_plots(pulsar_lle_datafile, 'pulsar_lle', plot_dir)


print('Intention pca results')
print_evaluation_stats(intention_pca_datafile)
print()

print('Intention ica results')
print_evaluation_stats(intention_ica_datafile, kmeans_clusters=3)
print()

print('Intention rp results')
print_evaluation_stats(intention_rp_datafile)
print()

print('Intention lle results')
print_evaluation_stats(intention_lle_datafile)
print()

print('pulsar pca results')
print_evaluation_stats(pulsar_pca_datafile)
print()

print('pulsar ica results')
print_evaluation_stats(pulsar_ica_datafile)
print()

print('pulsar rp results')
print_evaluation_stats(pulsar_rp_datafile)
print()

print('pulsar lle results')
print_evaluation_stats(pulsar_lle_datafile)
print()


# pulsar_PCA_X, pulsar_PCA_y = load_pulsar_PCA_reduced()
# intention_PCA_X, intention_PCA_y = load_intention_PCA_reduced()

# pulsar_ICA_X, pulsar_ICA_y = load_pulsar_ICA_reduced()
# intention_ICA_X, intention_ICA_y = load_intention_ICA_reduced()


# pulsar_RP_X, pulsar_RP_y = load_pulsar_RP_reduced()
# intention_RP_X, intention_RP_y = load_intention_RP_reduced()

# pulsar_LLE_X, pulsar_LLE_y = load_pulsar_LLE_reduced()
# intention_LLE_X, intention_LLE_y = load_intention_LLE_reduced()

# datafile = os.path.join(data_folder, "intention_pca_clustering.json")

# print_clustering_stats(intention_LLE_X, intention_LLE_y)
