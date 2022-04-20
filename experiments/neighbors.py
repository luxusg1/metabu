import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from metabu.utils import get_top_k_target
from metabu.learn_metafeatures import get_learned_metafeatures

from metafeatures.utils import get_dataset_metafeatures_train_test
from utils.hyperparameters import get_raw_priors, get_preprocessed_priors, preprocessed_df_priors, get_configspace
import json

datasets_has_priors = np.array(
    [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49, 53, 219, 2074, 2079, 3021,
     3022, 3481, 3549,
     3560, 3573, 3902, 3903, 3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971,
     9976, 9977, 9978,
     9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 125920, 125922, 146195, 146800,
     146817, 146819, 146820, 146821, 146824, 167125])


def get_neighbors_wrt_metabu_mf(test_task_id, classifier, cache_dir):
    top_k = 10
    ranking_column_name = 'predictive_accuracy'
    datasets_has_priors_use_for_train = [task_id for task_id in datasets_has_priors if
                                         task_id != test_task_id]

    # get preprocessed target representation
    target = get_raw_priors(os.path.join(cache_dir, f'data/{classifier}_best_all.arff'), classifier)
    _, target_preporcessor = get_preprocessed_priors(
        cache_dir=os.path.join(cache_dir, f'data/{classifier}_best_all.arff'),
        classifier=classifier, seed=43)
    cs = get_configspace(classifier=classifier, seed=43)
    preprocessed_target = preprocessed_df_priors(df=target, configspace=cs)
    preprocessed_target = pd.DataFrame(target_preporcessor.transform(preprocessed_target))

    if classifier == 'autosklearn':
        preprocessed_target['task_id'] = [configuration['task_id'] for configuration in target]
        preprocessed_target['predictive_accuracy'] = [configuration['predictive_performance'] for configuration in
                                                      target]
    else:
        preprocessed_target['task_id'] = target['task_id']
        preprocessed_target['predictive_accuracy'] = target['predictive_accuracy']

    # get topk preprocessed target
    top_k_target = get_top_k_target(preprocessed_target, preprocessed_target.task_id.unique(), top_k,
                                    ranking_column_name)

    # get preprocessed metafeatures
    train_mf, test_mf, augmented_data = get_dataset_metafeatures_train_test(test_task_id=test_task_id,
                                                                            datasets_has_priors_use_for_train=datasets_has_priors_use_for_train,
                                                                            cache_dir=cache_dir)

    # data_augmentation
    train_mf = pd.concat([train_mf, augmented_data], axis=0)

    # metabu_training
    model, metabu_mf = get_learned_metafeatures(
        datasets_has_priors_use_for_train=datasets_has_priors_use_for_train,
        train_metafeatures=train_mf, top_k_dataset=top_k_target,
        learning_rate=0.001,
        alpha=0.5,
        ranking_column_name=ranking_column_name)

    # get neighbors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_mf.values)
    x_test = torch.from_numpy(scaler.transform(test_mf.values)).to(device).float()
    x_train = torch.from_numpy(scaler.transform(train_mf.values)).to(device).float()

    test_metabu_mf = model(x_test).cpu().detach().numpy()
    train_metabu_mf = model(x_train[:len(datasets_has_priors_use_for_train)]).cpu().detach().numpy()

    distance_from_train = pairwise_distances(test_metabu_mf, train_metabu_mf, metric="l2")

    path_to_store = os.path.join(cache_dir, 'nearest',
                                 'general_statistical_info-theory_complexity_concept_itemset_clustering_landmarking_model-based')

    if not os.path.exists(path_to_store):
        os.makedirs(path_to_store)

    with open(os.path.join(path_to_store, "tid_{}_{}.json".format(test_task_id, classifier)), 'w') as jsonfile:
        jsonfile.write(str([datasets_has_priors_use_for_train[i] for i in
                            distance_from_train.argsort()[0]]))

    return [datasets_has_priors_use_for_train[i] for i in
            distance_from_train.argsort()[0]]
