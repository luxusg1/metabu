import os
import pandas as pd

from metabu.utils import get_cost_matrix

def get_basic_reprensetations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "basic_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]

def get_bootstrap_reprensetations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "bootstrap_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]

def get_target_reprensetations(pipeline, path):
    return pd.read_csv(os.path.join(path, "top_preprocessed_target_representation", pipeline.name + "_target_representation.csv"))


def run(cfg):
    print(get_basic_reprensetations(metafeature=cfg.metafeature, path=cfg.data_path))
    print(get_bootstrap_reprensetations(metafeature=cfg.metafeature, path=cfg.data_path))
    print(get_target_reprensetations(pipeline=cfg.pipeline, path=cfg.data_path))

    cost_matrix = get_cost_matrix(target_repr=target_reprs, task_ids=list_ids, verbose=False)