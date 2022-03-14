import argparse
import os
import pickle

import numpy as np
import openml
import pandas as pd
from pymfe.mfe import MFE


def get_metafeatures_dataset(dataset_id, group, only_download):
    dataset = openml.datasets.get_dataset(dataset_id)

    X, _, _, _ = dataset.get_data()
    y = X[dataset.default_target_attribute]
    X.drop([dataset.default_target_attribute], axis=1, inplace=True)
    X = X.apply(lambda x: x.fillna(x.value_counts().index[0]))
    X = X.values
    y = y.tolist()

    if only_download: return X, y


    mfe = MFE(groups=[group], measure_time="total", suppress_warnings=True)
    mfe.fit(X, y, suppress_warnings=True)
    ft = mfe.extract_with_confidence(sample_num=10)
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
    LIST_ALL_TASKS = [3, 6, 10, 11, 12, 14, 16, 18, 20, 21, 22, 23, 26, 28, 30, 31, 32, 36, 39, 41, 43, 44, 46, 50, 54, 59, 60, 61, 62, 151, 155, 161, 162, 164, 180, 181, 182, 183, 184, 187, 189, 209, 223, 225, 227, 230, 275, 277, 287, 292, 294, 298, 300, 307, 310, 312, 313, 329, 333, 334, 335, 336, 338, 339, 343, 346, 375, 377, 383, 385, 386, 387, 389, 391, 392, 395, 398, 400, 401, 444, 446, 448, 450, 457, 458, 461, 462, 463, 464, 465, 467, 468, 469, 472, 476, 477, 478, 479, 480, 679, 682, 685, 694, 713, 715, 716, 717, 718, 719, 720, 721, 722, 723, 725, 727, 728, 729, 730, 732, 734, 735, 737, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 751, 752, 754, 755, 756, 758, 759, 761, 762, 765, 766, 767, 768, 769, 770, 772, 775, 776, 777, 778, 779, 780, 782, 784, 785, 787, 788, 790, 791, 792, 793, 794, 795, 796, 797, 799, 801, 803, 804, 805, 806, 807, 808, 811, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 827, 828, 829, 830, 832, 833, 834, 835, 837, 841, 843, 845, 846, 847, 848, 849, 850, 853, 855, 857, 859, 860, 863, 864, 865, 866, 867, 868, 870, 871, 872, 873, 874, 875, 877, 878, 879, 880, 881, 882, 884, 885, 886, 889, 890, 892, 894, 895, 900, 901, 903, 904, 905, 910, 912, 913, 914, 915, 916, 917, 919, 921, 922, 923, 924, 925, 928, 932, 933, 934, 935, 936, 937, 938, 941, 942, 943, 946, 947, 950, 951, 952, 953, 954, 955, 956, 958, 959, 962, 964, 965, 969, 970, 971, 973, 974, 976, 977, 978, 979, 980, 983, 987, 988, 991, 994, 995, 997, 1004, 1005, 1006, 1009, 1011, 1013, 1014, 1015, 1016, 1019, 1020, 1021, 1022, 1025, 1026, 1036, 1038, 1040, 1041, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1075, 1079, 1081, 1082, 1104, 1106, 1107, 1115, 1116, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1129, 1131, 1132, 1133, 1135, 1136, 1137, 1140, 1141, 1143, 1144, 1145, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1160, 1162, 1163, 1165, 1167, 1169, 1217, 1236, 1237, 1238, 1413, 1441, 1442, 1443, 1444, 1446, 1448, 1449, 1450, 1451, 1452, 1454, 1455, 1457, 1459, 1460, 1464, 1467, 1471, 1475, 1481, 1482, 1486, 1488, 1489, 1496, 1498, 1500, 1501, 1505, 1507, 1508, 1509, 1510, 1516, 1517, 1519, 1520, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1544, 1545, 1546, 1556, 1557, 1561, 1562, 1563, 1564, 1565, 1567, 1568, 1569, 4134, 4135, 4153, 4340, 4534, 4538, 40474, 40475, 40476, 40477, 40478, 8, 37, 40, 48, 53, 197, 276, 278, 279, 285, 337, 384, 388, 394, 397, 459, 475, 683, 714, 724, 726, 731, 733, 736, 750, 753, 763, 764, 771, 773, 774, 783, 789, 800, 812, 825, 826, 836, 838, 851, 862, 869, 876, 887, 888, 891, 893, 896, 902, 906, 907, 908, 909, 911, 918, 920, 926, 927, 929, 931, 945, 948, 949, 996, 1012, 1054, 1071, 1073, 1077, 1078, 1080, 1084, 1100, 1117, 1159, 1164, 1412, 1447, 1453, 1472, 1473, 1483, 1487, 1503, 1512, 1513, 1518, 1543, 1600]
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
                # print("Error ", tid, " ", e)
                #if group == "info-theory":
                print(tid)
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
                        default="/home/tau/hrakotoa/code/priors_BO/data/metafeatures/openml_PMF",
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
            print(results, directory)
        else:
            import submitit

            executor = submitit.AutoExecutor(folder="tmp_slurm")
            executor.update_parameters(
                #tasks_per_node=1,  # one task per GPU
                #cpus_per_task=2,
                #nodes=1,
                timeout_min=30,
                mem_gb=15,
                name="Meta"
            )
            X, y = get_metafeatures_dataset(dataset_id=args.tid, group=args.group, only_download=True)
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
