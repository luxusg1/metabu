import numpy as np
import ot


def wasserstein_distance(id_a, id_b, top_k_dataset, ranking_column_name, return_map_matrix=False):
    dfs = top_k_dataset.loc[top_k_dataset.task_id == id_a]
    dft = top_k_dataset.loc[top_k_dataset.task_id == id_b]
    Xs = dfs.drop(['task_id', ranking_column_name], axis=1).values
    Xt = dft.drop(['task_id', ranking_column_name], axis=1).values

    M = ot.dist(Xs, Xt)
    M /= M.max()
    ns = len(Xs)
    nt = len(Xt)
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    G0 = ot.emd(a, b, M)

    if return_map_matrix: return (G0 * M).sum(), G0
    return (G0 * M).sum()


def get_cost_matrix_distribution(datasets_has_priors_use_for_train, top_k_dataset, ranking_column_name):
    print(f"Compute distance matrix between dataset w.r.t distribution of best target...")
    matrix_ot_distance = np.array(
        [[wasserstein_distance(id_a=a, id_b=b, top_k_dataset=top_k_dataset,
                               ranking_column_name=ranking_column_name) for a in
          datasets_has_priors_use_for_train] for b in
         datasets_has_priors_use_for_train])

    np.fill_diagonal(matrix_ot_distance, 0)

    matrix_ot_distance /= matrix_ot_distance.max()
    for i in range(matrix_ot_distance.shape[0]):
        for j in range(i):
            matrix_ot_distance[i, j] = matrix_ot_distance[j, i]

    # print(matrix_ot_distance)

    return matrix_ot_distance
