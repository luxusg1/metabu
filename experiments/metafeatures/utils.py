import os
import pickle

import numpy as np
import pandas as pd
from pymfe.mfe import MFE

from experiments.metafeatures.exctractor import normalize_metafeatures, get_possible_groups
from experiments.utils.metafeatures import MetaFeatures


def augment_dataset(task_id, metafeatures, metafeatures_confidence_interval, size):
    results_dict = {}
    if len(metafeatures_confidence_interval[task_id]) == 1:
        results_dict = [metafeatures.loc[task_id].to_dict()] * size
    for col, ci in metafeatures_confidence_interval[task_id].items():
        if col != "task_id":
            if isinstance(ci, np.ndarray) and np.isfinite(ci).all():
                # mean = (ci[1] - ci[0]) / 2 + ci[0]
                # sigma = ci[1] - mean
                results_dict[col] = np.random.uniform(ci[0], ci[1], size)
            else:
                results_dict[col] = [metafeatures.loc[task_id][col]] * size
    result_df = pd.DataFrame(results_dict)
    result_df["task_id"] = task_id
    return result_df.set_index("task_id")


def augment_datasets(list_task_id, metafeatures, metafeatures_confidence_interval, size):
    return pd.concat([augment_dataset(task_id, metafeatures, metafeatures_confidence_interval, size)
                      for task_id in list_task_id], axis=0)


def get_files(path, preprocessed=True):
    metafeatures = pd.read_csv(os.path.join(path, "openml_mfe.csv")).set_index("task_id")
    metafeatures_measure_time = pd.read_csv(os.path.join(path, "openml_mfe_measure_time.csv")).fillna(-1).set_index(
        "task_id")
    metafeatures_confidence_interval = pickle.load(
        open(os.path.join(path, "openml_mfe_confidence_interval.pkl"), "rb"))
    return normalize_metafeatures(
        metafeatures) if preprocessed else metafeatures, metafeatures_measure_time, metafeatures_confidence_interval


def get_PMF(path, preprocessed=True):
    metafeatures = pd.read_csv(os.path.join(path, "openml_mfe.csv")).set_index("task_id")
    metafeatures_measure_time = pd.read_csv(os.path.join(path, "openml_mfe_measure_time.csv")).fillna(-1).set_index(
        "task_id")
    metafeatures_confidence_interval = pickle.load(
        open(os.path.join(path, "openml_mfe_confidence_interval.pkl"), "rb"))
    return normalize_metafeatures(
        metafeatures) if preprocessed else metafeatures, metafeatures_measure_time, metafeatures_confidence_interval



def filter_by_group(df, groups):
    list_columns_to_select = get_features_by_group(groups)
    list_columns = df.columns
    return df[[col for col in list_columns if col.split(".")[0] in list_columns_to_select]]


def get_features_by_group(group):
    if isinstance(group, list):
        return MFE.valid_metafeatures(groups=group)
    return MFE.valid_metafeatures(groups=[group])


def get_handcrafted_metafeatures(path, subset, study="OpenML-CC18"):
    metafeatures = MetaFeatures(cache_directory=path, subset=subset)
    return metafeatures.get_all_by_study(study=study)


def get_dataset_metafeatures_train_test(test_task_id, datasets_has_priors_use_for_train, cache_dir,
                                        return_preprocessed=True):
    group_mfe = get_possible_groups()
    nb_augment = 1000

    metafeatures, metafeatures_measure_time, metafeatures_confidence_interval = get_files(
        path=os.path.join(cache_dir, "metafeatures"), preprocessed=False)
    augmented_data = augment_datasets(datasets_has_priors_use_for_train, metafeatures, metafeatures_confidence_interval,
                                      size=nb_augment)
    if not isinstance(test_task_id, list): test_task_id = [test_task_id]
    train_df = metafeatures.loc[datasets_has_priors_use_for_train]
    test_df = metafeatures.loc[test_task_id]
    augmented_data_test = augment_datasets(test_task_id, metafeatures, metafeatures_confidence_interval, size=100)
    train_df, test_df, augmented_data, augmented_data_test = filter_by_group(train_df, group_mfe), \
                                                             filter_by_group(test_df, group_mfe), \
                                                             filter_by_group(augmented_data, group_mfe), \
                                                             filter_by_group(augmented_data_test, group_mfe)

    if not return_preprocessed:
        return train_df, test_df, augmented_data

    _, (mean, std) = normalize_metafeatures(filter_by_group(train_df, group_mfe), return_moments=True)
    list_columns_wo_tid = [_ for _ in train_df.columns.tolist() if _ != "task_id"]

    range_values = 3

    train_df[list_columns_wo_tid] = np.clip((train_df[list_columns_wo_tid].values - mean) / (std + 1e-3),
                                            a_min=-range_values, a_max=range_values)
    test_df[list_columns_wo_tid] = np.clip((test_df[list_columns_wo_tid].values - mean) / (std + 1e-3),
                                           a_min=-range_values, a_max=range_values)
    augmented_data[list_columns_wo_tid] = np.clip((augmented_data[list_columns_wo_tid].values - mean) / (std + 1e-3),
                                                  a_min=-range_values, a_max=range_values)

    mf_baselines = get_handcrafted_metafeatures(path=cache_dir, subset="all").set_index("task_id")
    train_df_mfe = mf_baselines.loc[datasets_has_priors_use_for_train]
    test_df_mfe = mf_baselines.loc[test_task_id]

    _, (mean, std) = normalize_metafeatures(train_df_mfe, return_moments=True)
    list_columns_wo_tid = [_ for _ in train_df_mfe.columns.tolist() if _ != "task_id"]

    train_df_mfe[list_columns_wo_tid] = np.clip((train_df_mfe[list_columns_wo_tid].values - mean) / (std + 1e-3),
                                                a_min=-range_values, a_max=range_values)
    test_df_mfe[list_columns_wo_tid] = np.clip((test_df_mfe[list_columns_wo_tid].values - mean) / (std + 1e-3),
                                               a_min=-range_values, a_max=range_values)
    augmented_data_mfe = train_df_mfe.loc[train_df_mfe.index.repeat(nb_augment)]
    train_df = pd.concat([train_df, train_df_mfe], axis=1)
    test_df = pd.concat([test_df, test_df_mfe], axis=1)
    augmented_data = pd.concat([augmented_data, augmented_data_mfe], axis=1)

    to_use = [_ for _ in train_df.columns if ".sd" not in _]
    train_df, test_df, augmented_data = train_df[to_use].fillna(0), test_df[to_use].fillna(0), augmented_data[to_use].fillna(0)
    rename_col = [_.replace(".mean", "") for _ in train_df.columns]
    train_df.columns = rename_col
    test_df.columns = rename_col
    augmented_data.columns = rename_col

    col_to_drop = ["skewness", "c1", "class_ent", "kurtosis", "c2", "nr_cat", "nr_num", "nr_inst", "nr_class",
                   "num_to_cat", "eq_num_attr", "linear_discr", "naive_bayes", "one_nn", "random_node", "worst_node"]
    train_df.drop(col_to_drop, axis=1, inplace=True)
    test_df.drop(col_to_drop, axis=1, inplace=True)
    augmented_data.drop(col_to_drop, axis=1, inplace=True)
    to_use = [_ for _ in train_df.columns if ".sd" not in _]

    return train_df[to_use].fillna(0), test_df[to_use].fillna(0), augmented_data[to_use].fillna(0)
