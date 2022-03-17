import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from metabu.fused_gromov_wasserstein import train_fused_gromov_wasserstein
from metabu.target_representation import get_top_k_target_for_each_task, get_cost_matrix

'''
basic representation and target_representation are dataframe object and must have task_id column

'''


def train(basic_representation, target_representation, test_ids, train_ids, ranking_column_name):
    task_id_has_target_representation = target_representation.task_id.unique()

    if len(set(train_ids).intersection(set(test_ids))) > 0:
        raise ValueError('some of test_ids is also on train_ids')

    if set(train_ids) >= set(task_id_has_target_representation):
        raise ValueError('Some of train_ids have not a target representation')

    if basic_representation.isnull().values.any():
        raise ValueError('remove NaN value from basic representation')

    basic_representation_with_task_id_as_index = basic_representation.set_index('task_id')
    basic_representation_train = basic_representation_with_task_id_as_index.loc[train_ids]
    basic_representation_test = basic_representation_with_task_id_as_index.loc[test_ids]

    top_k_target_representation = get_top_k_target_for_each_task(target_representation=target_representation,
                                                                 task_ids=train_ids, k=10,
                                                                 ranking_column_name=ranking_column_name)

    cost_matrix = get_cost_matrix(target_representation=top_k_target_representation, task_ids=train_ids,
                                  ranking_column_name=ranking_column_name)

    idx_train = {id: np.where(basic_representation_with_task_id_as_index.index == id)[0] for id in train_ids}

    model, metabu_representation_train, metabu_representation_test = \
        train_fused_gromov_wasserstein(
            basic_representation_train=(basic_representation_with_task_id_as_index, idx_train),
            basic_representation_test=basic_representation_test,
            cost_matrix=cost_matrix,
            lr=0.001,
            seed=42,
            early_stopping=20,
            task_has_target_representation_used_for_train=train_ids)

    return model, metabu_representation_train, metabu_representation_test


def get_neighbors_wrt_metabu_mf(basic_representation, target_representation, test_ids, train_ids, ranking_column_name):
    _, metabu_representation_train, metabu_representation_test = train(basic_representation, target_representation,
                                                                       test_ids, train_ids, ranking_column_name)

    neighbors = []
    for elements in metabu_representation_test:
        distance_from_train = pairwise_distances([elements], metabu_representation_train, metric="l2")
        neighbors.append([train_ids[i] for i in distance_from_train.argsort()[0]])

    return neighbors


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--basic_representation_file',
                        default='../examples/data/basic_representation.csv',
                        help='basic representation file')

    parser.add_argument('--target_representation_file',
                        required=False,
                        default='../examples/data/target_representation_adaboost.csv',
                        help='target representation file')

    parser.add_argument('--ranking_column_name',
                        required=True,
                        help='the name of columns to rank the target representation in the target representation file')

    parser.add_argument('--store',
                        default="examples/output",
                        help='Path to store')

    parser.add_argument('--test_ids', nargs='+',
                        help='list of task id to be removed when train metabu_representation',
                        required=True)

    args = parser.parse_args()

    # get basic representation
    basic_representation = pd.read_csv(args.basic_representation_file)
    basic_representation = basic_representation.fillna(0)

    # get target representation
    target_representation = pd.read_csv(args.target_representation_file)

    # train metabu representation
    test_ids = [int(task_id) for task_id in args.test_ids]
    train_ids = [task_id for task_id in list(target_representation.task_id.unique()) if task_id not in test_ids]
    print(train_ids)
    _, metabu_representation_train, metabu_representation_test = train(basic_representation=basic_representation,
                                                                       target_representation=target_representation,
                                                                       test_ids=test_ids,
                                                                       train_ids=train_ids,
                                                                       ranking_column_name=args.ranking_column_name)

    # store metabu mf
    metabu_representation_test = pd.DataFrame(metabu_representation_test)
    metabu_representation_test['task_id'] = args.test_ids
    metabu_representation_test.to_csv(os.path.join(args.store, 'metabu_representation.csv'), index=False)
