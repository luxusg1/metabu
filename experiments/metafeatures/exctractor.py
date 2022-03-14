import argparse
import os
import pickle

import numpy as np
import openml
import pandas as pd
from pymfe.mfe import MFE


def get_metafeatures_dataset(task_id, group, only_download):
    task = openml.tasks.get_task(task_id)
    split = task.get_train_test_split_indices()
    if task_id == 167125:
        X, y = task.get_X_and_y()
        X, y = X[split[0]], y[split[0]]
    else:
        X, y = task.get_X_and_y(dataset_format="pandas")
        X_train, y_train = X.iloc[split[0]], y[split[0]]
        if task_id == 3021: X_train.drop("TBG", axis=1, inplace=True)
        X_train = X_train.apply(lambda x: x.fillna(x.value_counts().index[0]))
        X = X_train.values
        y = y_train.tolist()
    if only_download: return X, y
    # , "general", "statistical", "info-theory", "complexity", "concept", "clustering",
    #                       "landmarking", "model-based"

    mfe = MFE(groups=[group], measure_time="total", suppress_warnings=True)
    mfe.fit(X, y, suppress_warnings=True)
    ft = mfe.extract_with_confidence(sample_num=20)
    return ft


def get_metafeature(X, y, metafeatures_name):
    mfe = MFE(features=[metafeatures_name], measure_time="total", suppress_warnings=True)
    mfe.fit(X, y, suppress_warnings=True)
    ft = mfe.extract_with_confidence(sample_num=20)
    return ft


def get_possible_groups():
    return ["general", "statistical", "info-theory", "complexity", "concept", "clustering", "landmarking",
            "model-based", "itemset"]


def generate_dataset(proprety):
    has_error = False
    groups = get_possible_groups()
    LIST_ALL_TASKS = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49, 53, 219, 2074, 2079,
                      3021, 3022, 3481, 3549, 3560, 3573, 3902, 3903, 3904, 3913, 3917, 3918, 7592, 9910, 9946,
                      9952, 9957, 9960, 9964, 9971, 9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965,
                      14969, 14970, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821, 146822, 146824,
                      146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141]
    metafeatures_dict = {id: {} for id in LIST_ALL_TASKS}
    for group in groups:
        directory = os.path.join(args.path, group)
        print("#" * 5, "\t", group, "\t", "#" * 5)
        for tid in LIST_ALL_TASKS:
            try:
                with open(os.path.join(directory, f"mfe_{tid}.csv"), "rb") as input_file:
                    res = pickle.load(input_file)
                    result = {name: val[proprety] for name, val in res.items()}
                    result["task_id"] = tid
                    metafeatures_dict[tid] = {**result, **(metafeatures_dict[tid])}
            except Exception as e:
                print("Error ", tid, " ", e)
                has_error = True
    return metafeatures_dict, has_error


def normalize_metafeatures(df, return_moments=False):
    list_columns = list(df.columns)
    list_columns_wo_tid = [_ for _ in list_columns if _ != "task_id"]
    X = df[list_columns_wo_tid].values
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    if return_moments:
        return df.fillna(0), (mean, std)
    df[list_columns_wo_tid] = (X - mean) / (std + 1e-3)
    return df.fillna(0)


def augment_dataset(task_id, metafeatures, metafeatures_confidence_interval, size):
    results_dict = {}
    for col, ci in metafeatures_confidence_interval[task_id].items():
        if col != "task_id":
            if isinstance(ci, np.ndarray) and np.isfinite(ci).all():
                mean = (ci[1] - ci[0]) / 2 + ci[0]
                sigma = ci[1] - mean
                results_dict[col] = np.random.uniform(mean - sigma, mean + sigma, size)
            else:
                results_dict[col] = [metafeatures.loc[task_id][col]] * size
    result_df = pd.DataFrame(results_dict)
    result_df["task_id"] = task_id
    return result_df.set_index("task_id")


def augment_datasets(list_task_id, metafeatures, metafeatures_confidence_interval, size):
    return pd.concat([augment_dataset(task_id, metafeatures, metafeatures_confidence_interval, size)
                      for task_id in list_task_id], axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--use_slurm', action='store_true')
    parser.add_argument('--tid',
                        type=int,
                        help='Task id')
    parser.add_argument('--group',
                        required=False,
                        help='Group of metafeatures: https://pymfe.readthedocs.io/en/latest/auto_pages/meta_features_description.html')
    parser.add_argument('--path',
                        default="/home/tau/hrakotoa/code/metabu/data/metafeatures/openml",
                        help='Path to store')

    args = parser.parse_args()

    if not args.aggregate:
        directory = os.path.join(args.path, args.group)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not args.use_slurm:
            ft = get_metafeatures_dataset(args.tid, args.group, args.download)
            results = {name: {"val": val, "measure_time": measure_time, "confidence_interval": confidence_interval} for
                       (name, val, measure_time, confidence_interval) in zip(ft[0], ft[1], ft[2], ft[3])}
        else:
            import submitit

            executor = submitit.AutoExecutor(folder="tmp_slurm")
            executor.update_parameters(
                tasks_per_node=1,  # one task per GPU
                cpus_per_task=2,
                nodes=1,
                timeout_min=30,
                mem_gb=15,
                name="Meta"
            )
            X, y = get_metafeatures_dataset(task_id=args.tid, group=args.group, only_download=True)
            mtfs_subset = list(MFE.valid_metafeatures(groups=[args.group]))
            print(mtfs_subset)
            jobs = [executor.submit(get_metafeature, X, y, metafeaure) for metafeaure in mtfs_subset]
            results = []
            for job in jobs:
                try:
                    res = job.result()
                    name, val, measure_time, confidence_interval = res
                    res = {"val": val[0], "measure_time": measure_time[0],
                           "confidence_interval": confidence_interval[0]}
                    results.append(res)
                except Exception as e:
                    results.append(None)

            results = {name: val_dict for name, val_dict in zip(mtfs_subset, results) if val_dict is not None}

        with open(os.path.join(directory, f"mfe_{args.tid}.csv"), "wb") as output_file:
            pickle.dump(results, output_file)
    else:
        metafeatures_dict, has_error = generate_dataset("val")
        metafeatures_dict_time, has_error_time = generate_dataset("measure_time")
        metafeatures_dict_confidence_interval, has_error_confidence_interval = generate_dataset("confidence_interval")
        if not has_error and not has_error_time:
            df = pd.DataFrame(metafeatures_dict.values())
            df_time = pd.DataFrame(metafeatures_dict_time.values())
            df.to_csv(os.path.join(args.path, "openml_mfe.csv"), index=None)
            df_time.to_csv(os.path.join(args.path, "openml_mfe_measure_time.csv"), index=None)
            pickle.dump(metafeatures_dict_confidence_interval,
                        open(os.path.join(args.path, "openml_mfe_confidence_interval.pkl"), "wb"))
            print("Done: saved at ", os.path.join(args.path, "openml_mfe.csv"))

## sbatch --ntasks=1 --cpus-per-task=5 --hint=nomultithread --mem=30GB --time=30:50:00 --account=cpa@cpu  compute_metafeatures.sh
