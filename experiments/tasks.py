import os
import numpy as np
import pandas as pd

from metabu import Metabu

from sklearn.metrics import pairwise_distances
from metabu.utils import get_cost_matrix, get_ndcg_score


def get_basic_representations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "basic_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]


def get_bootstrap_representations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "bootstrap_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]


def get_target_representations(pipeline, path):
    return pd.read_csv(
        os.path.join(path, "top_preprocessed_target_representation", pipeline.name + "_target_representation.csv"))


def get_metabu_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids):
    basic_reprs["boostrap"] = 0

    bootstrap_reprs = get_bootstrap_representations(metafeature=cfg.metafeature, path=cfg.data_path)
    bootstrap_reprs = bootstrap_reprs[bootstrap_reprs.task_id.isin(list_ids)]
    bootstrap_reprs["boostrap"] = 1

    combined_basic_reprs = pd.concat([basic_reprs, bootstrap_reprs], axis=0)
    combined_basic_reprs = pd.concat([
        combined_basic_reprs[combined_basic_reprs.task_id.isin(train_ids)],
        combined_basic_reprs[combined_basic_reprs.task_id.isin(test_ids)]
    ], axis=0)

    metabu = Metabu(verbose=False, seed=cfg.seed)
    repr_train, repr_test = metabu.train_predict(
        basic_reprs=combined_basic_reprs.drop(["boostrap"], axis=1),
        target_reprs=target_reprs,
        column_id="task_id",
        train_ids=train_ids,
        test_ids=test_ids
    )
    metabu_reprs = np.concatenate([repr_train, repr_test], axis=0)
    metabu_reprs = pd.DataFrame(metabu_reprs, columns=[f"col{_}" for _ in range(metabu_reprs.shape[1])])
    metabu_reprs["task_id"] = combined_basic_reprs["task_id"].values
    metabu_reprs["boostrap"] = combined_basic_reprs["boostrap"].values
    return metabu_reprs[metabu_reprs.boostrap == 0].drop(["boostrap"], axis=1)


def run_task1(cfg):
    target_reprs = get_target_representations(pipeline=cfg.pipeline, path=cfg.data_path)
    list_ids = sorted(list(target_reprs["task_id"].unique()))

    if cfg.openml_tid not in list_ids:
        raise Exception(f"OpenML task {cfg.openml_tid} does not have target representations.")

    basic_reprs = get_basic_representations(metafeature=cfg.metafeature, path=cfg.data_path)
    basic_reprs = basic_reprs[basic_reprs.task_id.isin(list_ids)]

    if cfg.metafeature.name == "metabu":
        train_ids = [_ for _ in list_ids if _ != cfg.openml_tid]
        test_ids = [cfg.openml_tid]

        basic_reprs = get_metabu_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids)

    basic_reprs = basic_reprs.set_index("task_id")

    true_dist = get_cost_matrix(target_repr=target_reprs, task_ids=list_ids, verbose=False)
    pred_dist = pairwise_distances(basic_reprs.loc[list_ids])

    id_test = list_ids.index(cfg.openml_tid)

    print("Task 1: \n- pipeline: {0} \n- Metafeature: {1} \n- OpenML task: {3} \n- NDCG@{2}: {4}".format(
        cfg.pipeline.name,
        cfg.metafeature.name,
        cfg.task.ndcg,
        cfg.openml_tid,
        get_ndcg_score(dist_pred=np.array([pred_dist[id_test]]), dist_true=np.array([true_dist[id_test]]),
                       k=cfg.task.ndcg)
    ))

    if cfg.output_file is not None:
        with open(cfg.output_file, 'a') as the_file:
            the_file.write("{0},{1},{2},{3},{4}\n".format(
                cfg.pipeline.name,
                cfg.metafeature.name,
                cfg.openml_tid,
                cfg.task.ndcg,
                get_ndcg_score(dist_pred=np.array([pred_dist[id_test]]), dist_true=np.array([true_dist[id_test]]),
                               k=cfg.task.ndcg)
            ))


