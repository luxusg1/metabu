from metabu.utils import get_significative_best, wasserstein_distance
import numpy as np
import pandas as pd


def get_cost_matrix(target_representation, task_ids, ranking_column_name):
    print(f"Compute distance matrix between datasets distribution wrt target representation ...")
    matrix_ot_distance = []
    for task_a in task_ids:
        temp_distance = []
        for task_b in task_ids:
            target_representation_for_task_a = target_representation.loc[target_representation.task_id == task_a].drop(
                ['task_id', ranking_column_name], axis=1).values
            target_representation_for_task_b = target_representation.loc[target_representation.task_id == task_b].drop(
                ['task_id', ranking_column_name], axis=1).values
            target_distance_between_task_a_b = wasserstein_distance(target_representation_for_task_a,
                                                                    target_representation_for_task_b)
            temp_distance.append(target_distance_between_task_a_b)
        matrix_ot_distance.append(temp_distance)

    matrix_ot_distance = np.array(matrix_ot_distance)
    np.fill_diagonal(matrix_ot_distance, 0)
    matrix_ot_distance /= matrix_ot_distance.max()
    for i in range(matrix_ot_distance.shape[0]):
        for j in range(i):
            matrix_ot_distance[i, j] = matrix_ot_distance[j, i]

    return matrix_ot_distance


def get_top_k_target_for_each_task(target_representation, task_ids, k, ranking_column_name):
    top_k_target_representation = []
    for it, tid in enumerate(task_ids):
        tmp = get_significative_best(target_representation[target_representation.task_id == tid].copy(), k,
                                     ranking_column_name)
        if tmp.shape[0] == 0: raise Exception(
            "Do not have enough sampling on dataset tid:{}. Only {} samples".format(tid, tmp.shape[0]))
        top_k_target_representation.append(tmp)
    return pd.concat(top_k_target_representation, axis=0)
