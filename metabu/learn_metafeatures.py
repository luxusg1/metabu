from metabu import train_fused_gromov_wasserstein, get_cost_matrix_distribution
import numpy as np


def get_learned_metafeatures(datasets_has_priors_use_for_train, train_metafeatures, top_k_dataset,
                             learning_rate, alpha, ranking_column_name, seed=1):
    idx_train = {id: np.where(train_metafeatures.index == id)[0] for id in datasets_has_priors_use_for_train}
    cost_matrix = get_cost_matrix_distribution(datasets_has_priors_use_for_train=datasets_has_priors_use_for_train,
                                               top_k_dataset=top_k_dataset, ranking_column_name=ranking_column_name)

    model, train_learned_mfs, _ = train_fused_gromov_wasserstein(
        data_train=(train_metafeatures, idx_train),
        early_stopping=20,
        cost_matrix=cost_matrix,
        seed=seed,
        lr=learning_rate,
        alpha=alpha,
        datasets_has_priors_use_for_train=datasets_has_priors_use_for_train)

    return model, train_learned_mfs
