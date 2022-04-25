import math
import os

import numpy as np
import pandas as pd
from ConfigSpace.read_and_write import json
from sklearn.metrics import pairwise_distances

from metabu import Metabu
from metabu.utils import get_cost_matrix, get_ndcg_score

from .smac_wrapper import Runner

def get_basic_representations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "basic_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]


def get_bootstrap_representations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "bootstrap_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]


def get_target_representations(pipeline, path):
    return pd.read_csv(
        os.path.join(path, "top_preprocessed_target_representation", pipeline.name + "_target_representation.csv"))


def get_raw_target_representations(pipeline, path):
    c = {"True": True, "False": False}
    df_hp= pd.read_csv(
        os.path.join(path, "top_raw_target_representation", pipeline + "_target_representation.csv")).drop(["predictive_accuracy"], axis=1)
    if pipeline == "random_forest":
        df_hp.classifier__bootstrap = df_hp.classifier__bootstrap.map(c).astype(bool)
    if pipeline == "adaboost":
        for columns_name in ['classifier__algorithm', 'imputation__strategy']:
            df_hp[columns_name] = df_hp[columns_name].str.decode('utf-8')
        df_hp.classifier__algorithm = np.where((df_hp.classifier__algorithm == 'SAMME_R'),
                                               'SAMME.R', df_hp.classifier__algorithm)
        df_hp = df_hp.astype({"classifier__base_estimator__max_depth": int,
                              "classifier__learning_rate": float,
                              "classifier__n_estimators": int})

    if pipeline == "libsvm_svc":
        for columns_name in ['classifier__kernel', 'imputation__strategy']:
            df_hp[columns_name] = df_hp[columns_name].str.decode('utf-8')

        df_hp = df_hp.astype({"classifier__C": float,
                              "classifier__degree": int,
                              "classifier__gamma": float,
                              "classifier__coef0": float,
                              "classifier__tol": float,
                              "classifier__max_iter": int,
                              "classifier__shrinking": bool})

    return df_hp

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

    metabu = Metabu(alpha=0.5,
                    lambda_reg=1e-3,
                    learning_rate=0.01,
                    early_stopping_patience=20,
                    early_stopping_criterion_ndcg=cfg.task.ndcg,
                    verbose=False,
                    seed=cfg.seed)
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


def get_configuration_space(cfg):
    with open(os.path.join(cfg.data_path, "configspace", "{}_configspace.json".format(cfg.pipeline.name)), 'r') as fh:
        json_string = fh.read()
        return json.read(json_string)


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


def run_task2(cfg):
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

    pred_dist = pairwise_distances(basic_reprs.loc[list_ids])
    id_test = list_ids.index(cfg.openml_tid)

    # Get id neighbors
    id_neighbors = [list_ids[_] for _ in pred_dist[id_test].argsort()[1:]][:cfg.task.ndcg]

    hps = get_raw_target_representations(pipeline=cfg.pipeline.name, path=cfg.data_path)
    hps = hps[hps.task_id.isin(id_neighbors)]
    hps["weights"] = hps.task_id.map(lambda x: math.exp(-id_neighbors.index(x)))
    hps["weights"] /= hps["weights"].sum()

    idx = np.random.choice(hps.index, cfg.task.nb_iterations, p=hps.weights, replace=False)
    cs = get_configuration_space(cfg)

    run = Runner(pipeline=cfg.pipeline.name, config_space=cs, seed=cfg.seed)
    results = run.exec(task_id=cfg.openml_tid, hps=hps.drop(["task_id", "weights"], axis=1).loc[idx].to_dict(orient="records"), counter=0)

    print("Results with task_id:{}, meta-feature={}, and pipeline={}".format(results[0]["task_id"],
                                                                             cfg.pipeline.name,
                                                                             results[0]["pipeline"]))
    for res in results:
        print("Iter={}\n\t hp={}\n\t perf={}".format(res["hp_id"] + 1, res["hp"], res["performance"]))


