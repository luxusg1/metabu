import math
from multiprocessing import Pool

import numpy as np
import ot
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def wasserstein_distance(distributions, return_map_matrix=False):
    distribution_a, distribution_b = distributions
    M = ot.dist(distribution_a, distribution_b)
    M /= M.max()
    ns = len(distribution_a)
    nt = len(distribution_b)
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    G0 = ot.emd(a, b, M)

    if return_map_matrix: return (G0 * M).sum(), G0
    return (G0 * M).sum()


def intrinsic_estimator(matrix_distance):
    muL = []
    N = len(matrix_distance)
    Femp = []
    for i in range(len(matrix_distance)):
        distances_ = np.unique(matrix_distance[i])
        NN = np.argsort(distances_)[1:3]
        first = NN[0]
        second = NN[1]
        mu_i = distances_[second] / (distances_[first] + (10 ** (-3)))
        muL.append(mu_i)
    muL = np.sort(muL)
    cutoff = int(np.floor(0.9 * len(muL)))
    muL = muL[0:cutoff + 1]
    muL = [x if x > 0 else 1 + 10 ** (-3) for x in muL]
    muL = np.asarray([math.log(mu_i) for mu_i in muL]).reshape(-1, 1)
    step = 1 / N
    Femp = [i * step for i in range(1, len(muL) + 1)]
    Femp = np.asarray([-math.log(1 - x) for x in Femp]).reshape(-1, 1)
    clf = LinearRegression(fit_intercept=False)
    clf.fit(muL, Femp)
    intrinsic = clf.coef_[0][0]
    return math.ceil(intrinsic)


def get_cost_matrix(target_repr, task_ids, verbose, column_id, n_cpus=1):
    matrix_ot_distance = []
    for task_a in tqdm(task_ids, disable=not verbose):
        temp_distance = []
        p = Pool(n_cpus)
        params = [
            (target_repr.loc[target_repr[column_id] == task_a].drop(
                [column_id], axis=1).values,
             target_repr.loc[target_repr[column_id] == task_b].drop(
                 [column_id], axis=1).values
             ) for task_b in task_ids
        ]
        temp_distance = p.map(wasserstein_distance, params)

        # for task_b in task_ids:
        #     target_representation_for_task_a = target_repr.loc[target_repr.task_id == task_a].drop(
        #         ['task_id'], axis=1).values
        #     target_representation_for_task_b = target_repr.loc[target_repr.task_id == task_b].drop(
        #         ['task_id'], axis=1).values
        #     target_distance_between_task_a_b = wasserstein_distance(target_representation_for_task_a,
        #                                                             target_representation_for_task_b)
        #     temp_distance.append(target_distance_between_task_a_b)
        matrix_ot_distance.append(temp_distance)

    matrix_ot_distance = np.array(matrix_ot_distance)
    np.fill_diagonal(matrix_ot_distance, 0)
    matrix_ot_distance /= matrix_ot_distance.max()
    for i in range(matrix_ot_distance.shape[0]):
        for j in range(i):
            matrix_ot_distance[i, j] = matrix_ot_distance[j, i]

    return matrix_ot_distance


def get_ndcg_score(dist_pred, dist_true, k=10):
    pred_rank = dist_pred.argsort().argsort()
    true_rank = dist_true.argsort().argsort()

    pred_rank[np.where(pred_rank < k)] = 1
    pred_rank[np.where(pred_rank >= k)] = 0
    true_rank[np.where(true_rank < k)] = 1
    true_rank[np.where(true_rank >= k)] = 0

    return ndcg_score(y_true=true_rank, y_score=pred_rank, k=k)


def get_pca_importances(data):
    data_scaled = StandardScaler().fit_transform(data)
    pca = PCA()
    pca.fit_transform(data_scaled)
    return np.abs(pca.components_[0])
