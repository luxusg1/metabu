import numpy as np
import pandas as pd
import ot


def wasserstein_distance(id_a, id_b, top_k_dataset, prior_preprocessor, return_map_matrix=False):
    if isinstance(top_k_dataset, pd.DataFrame):
        dfs = top_k_dataset.loc[top_k_dataset.task_id == id_a]
        dft = top_k_dataset.loc[top_k_dataset.task_id == id_b]
        ys = dfs["predictive_accuracy"].values
        yt = dft["predictive_accuracy"].values
    else:
        dfs = [c for c in top_k_dataset if c["task_id"] == id_a]
        dft = [c for c in top_k_dataset if c["task_id"] == id_b]

        ys = np.array([c["predictive_performance"] for c in dfs])
        yt = np.array([c["predictive_performance"] for c in dft])

    Xs = prior_preprocessor.transform(dfs)
    Xt = prior_preprocessor.transform(dft)

    M = ot.dist(Xs, Xt)
    M /= M.max()
    ns = len(Xs)
    nt = len(Xt)
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    G0 = ot.emd(a, b, M)

    if return_map_matrix: return (G0 * M).sum(), G0
    return (G0 * M).sum()


def get_cost_matrix_distribution(datasets_has_priors_use_for_train, top_k_dataset, prior_preprocessor):

    print(f"Compute distance matrix between dataset w.r.t distribution of best target...")
    matrix_ot_distance = np.array(
        [[wasserstein_distance(id_a=a, id_b=b, top_k_dataset=top_k_dataset,
                               prior_preprocessor=prior_preprocessor) for a in
          datasets_has_priors_use_for_train] for b in
         datasets_has_priors_use_for_train])

    np.fill_diagonal(matrix_ot_distance, 0)

    matrix_ot_distance /= matrix_ot_distance.max()
    for i in range(matrix_ot_distance.shape[0]):
        for j in range(i):
            matrix_ot_distance[i, j] = matrix_ot_distance[j, i]

    print(matrix_ot_distance)

    return matrix_ot_distance
