import collections
import json
import math
import os
import pickle

import ConfigSpace
import numpy as np
import openml
import pandas as pd
import sklearn
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import sys

sys.path.append('..')

import openmlpimp
from openmlpimp.utils import modeltype_to_classifier
from openmlstudy14.preprocessing import ConditionalImputer
from utils.hyperparameters import get_top_k_raw_priors, datasets_has_priors


def intrinsicEstimator(matrix_distance):
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


def integer_encode_dataframe(df: pd.DataFrame,
                             config_space: ConfigSpace.ConfigurationSpace) -> pd.DataFrame:
    """
    Takes a dataframe and encodes all columns whose name aligns with a
    Constant,  UnparameterzedHyperparameter or CategoricalHyperparameter as
    integer.

    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to encode

    config_space: ConfigSpace.ConfigurationSpace
        the configuration space. Each hyperparameter that has a name
        corresponding with a dataframe column will be encoded appropriately.

    Returns
    -------
    pd.DataFrame
        a dataframe with (for all hyperparameters) integer encoded variables
    """
    mask = df.applymap(type) != bool
    d = {True: 'True', False: 'False'}
    df = df.where(mask, df.replace(d))
    for column_name in df.columns.values:
        if column_name in config_space.get_hyperparameter_names():
            hyperparameter = config_space.get_hyperparameter(column_name)
            if isinstance(hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter):
                # numeric hyperparameter, don't do anything
                pass
            elif isinstance(hyperparameter, (ConfigSpace.hyperparameters.Constant,
                                             ConfigSpace.hyperparameters.UnParametrizedHyperparameter)):
                # encode as constant value. can be retrieved from config space later
                df[column_name] = 0
                df[column_name] = pd.to_numeric(df[column_name])
            elif isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                df[column_name] = df[column_name].apply(lambda x: hyperparameter.choices.index(x))
                df[column_name] = pd.to_numeric(df[column_name])
            else:
                raise NotImplementedError('Function not implemented for '
                                          'Hyperparameter: %s' % type(hyperparameter))
    return df


def get_suite(cache_dir=os.path.expanduser("~/experiments"), study="OpenML-CC18"):
    cache_suite_dir = os.path.join(cache_dir, 'openml')
    path_suite_file = os.path.join(cache_suite_dir, study + '.pkl')

    if not os.path.isfile(path_suite_file):
        if not os.path.isdir(cache_suite_dir):
            os.makedirs(cache_suite_dir)
        suite = openml.study.get_suite(study)
        with open(path_suite_file, 'wb') as f:
            pickle.dump(suite, f, pickle.HIGHEST_PROTOCOL)
    with open(path_suite_file, 'rb') as f:
        suite = pickle.load(f)

    return suite


def cache_setups(cache_directory, flow_id, bestN):
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    setups = openml.setups.list_setups(flow=flow_id)
    with open(cache_directory + '/setup_list_best%d.pkl' % bestN, 'wb') as f:
        pickle.dump(setups, f, pickle.HIGHEST_PROTOCOL)


def cache_task_setup_scores(cache_directory, study, flow_id):
    # print(setups.keys())
    task_setup_scores = collections.defaultdict(dict)
    for task_id in study.tasks:
        runs = openml.evaluations.list_evaluations("predictive_accuracy", tasks=[task_id], flows=[flow_id])
        for run in runs.values():
            task_setup_scores[task_id][run.setup_id] = run.value
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    with open(cache_directory + '/' + study.alias + 'best_setup_per_task.pkl', 'wb') as f:
        pickle.dump(task_setup_scores, f, pickle.HIGHEST_PROTOCOL)


def get_setup(flow_id, study, bestN=10, cache_dir=os.path.expanduser("~/experiments")):
    cache_directory = cache_dir + '/cache_kde/' + str(flow_id) + '/vanilla'
    setups_scores_cache_file = cache_directory + '/' + study + 'best_setup_per_task.pkl'
    setups_cache_file = cache_directory + '/setup_list_best%d.pkl' % bestN

    if not os.path.isfile(setups_cache_file):
        print('%s No cache file for setups (expected: %s), will create one ... ' % (
            openmlpimp.utils.get_time(), setups_cache_file))
        cache_setups(cache_directory, flow_id, bestN)
        print('%s Cache created. Available in: %s' % (openmlpimp.utils.get_time(), setups_cache_file))

    with open(setups_cache_file, 'rb') as f:
        setups = pickle.load(f)

    if not os.path.isfile(setups_scores_cache_file):
        print('%s No cache file for task setup scores (expected: %s), will create one ... ' % (
            openmlpimp.utils.get_time(), setups_scores_cache_file))
        suite = get_suite(study=study, cache_dir=cache_dir)
        cache_task_setup_scores(cache_directory, suite, flow_id)
        print('%s Cache created. Available in: %s' % (openmlpimp.utils.get_time(), setups_scores_cache_file))

    with open(setups_scores_cache_file, 'rb') as f:
        task_setup_scores = pickle.load(f)

    return setups, task_setup_scores


def get_fast_task(suite, slow_limits):
    fast_task = []
    for task in suite.tasks:
        if task not in [167124]:
            X_df, Y_df = openml.tasks.get_task(task).get_X_and_y()
            if (len(X_df) * 0.8) < slow_limits:
                fast_task.append(task)
    return fast_task


def split_list(split_part_len, test_list):
    size = len(test_list)
    idx_list = [idx + 1 for idx, val in
                enumerate(test_list) if (idx + 1) % split_part_len == 0]

    res = [test_list[i: j] for i, j in
           zip([0] + idx_list, idx_list +
               ([size] if idx_list[-1] != size else []))]
    return res


def get_nominal_index(df_data):
    indices_cat = []
    indices_num = []
    for i in range(len(df_data.dtypes) - 1):
        if str(df_data.dtypes[i]) == 'float64' or str(df_data.dtypes[i]) == 'int64':
            indices_num.append(i)
        elif str(df_data.dtypes[i]) == 'category' or str(df_data.dtypes[i] == 'object'):
            indices_cat.append(i)
        else:
            print(df_data.dtypes[i])
    return indices_cat, indices_num


# normalize all columns not in holdout
def normalize_dataframe(df, holdout, index):
    df_normalize = df.drop(holdout, axis=1)
    mf_values = df_normalize.values
    N = mf_values.shape[0]
    mf_normalized = []
    # for i in range(N):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(mf_values[np.arange(N) != index])
    for i in range(N):
        mf_normalized.append(min_max_scaler.transform([mf_values[i]])[0])

    df_normalize = pd.DataFrame(np.array(mf_normalized), columns=df_normalize.columns,
                                index=df_normalize.index)

    for columns_name in holdout:
        df_normalize[columns_name] = df[columns_name]

    return df_normalize.reindex(columns=df.columns)


def configspace_to_json(configspace: ConfigSpace.ConfigurationSpace, df_priors):
    input_parameters = {}
    hp_names = configspace.get_hyperparameter_names()
    for hp_name in hp_names:
        hp = configspace.get_hyperparameter(hp_name)
        input_parameters[hp_name] = {}

        if isinstance(hp, UniformIntegerHyperparameter):

            input_parameters[hp_name]["parameter_type"] = 'integer'
            input_parameters[hp_name]["values"] = [hp.lower, hp.upper]
            input_parameters[hp_name]["parameter_default"] = hp.default_value
            input_parameters[hp_name]["prior"] = "uniform"


        elif isinstance(hp, UniformFloatHyperparameter):

            input_parameters[hp_name]["parameter_type"] = 'real'
            input_parameters[hp_name]["values"] = [hp.lower, hp.upper]
            input_parameters[hp_name]["parameter_default"] = hp.default_value
            input_parameters[hp_name]["prior"] = "gaussian"

        elif isinstance(hp, CategoricalHyperparameter):

            input_parameters[hp_name]["parameter_type"] = 'integer'
            input_parameters[hp_name]["values"] = [int(df_priors[hp_name].min()), int(df_priors[hp_name].max())]
            input_parameters[hp_name]["parameter_default"] = int(df_priors[hp_name].min())
            input_parameters[hp_name]["prior"] = "uniform"

    return input_parameters


def generate_hyperMapper_json(task_id, configspace, classifier, cache_dir, nb_datasets=15, k=50):
    priors_data_path = os.path.join(cache_dir, "result",
                                    f"tid_{task_id}_{classifier}_k_{k}_nb_datasets_{nb_datasets}.csv")
    if not os.path.isdir(os.path.join(cache_dir, "result")):
        os.mkdir(os.path.join(cache_dir, "result"))
    priors_data = pd.DataFrame()
    # if not os.path.isfile(priors_data_path):
    with open(os.path.join(cache_dir, "nearest", "train_fused_gromov_wasserstein",
                           "tid_{}_{}.json".format(task_id, classifier)), 'r') as json_file:
        nearest_task = json.load(json_file)

    nearest_task = [dt for dt in nearest_task if dt in datasets_has_priors][:nb_datasets]
    path_prios_file = os.path.join(cache_dir, "data", f"{classifier}_best_500.arff")
    top_k_raw_priors = get_top_k_raw_priors(cache_dir=path_prios_file, classifier=classifier, list_tid=nearest_task,
                                            k=k)
    priors_data = top_k_raw_priors
    from sklearn.preprocessing import LabelEncoder
    categorical_columns_data = priors_data.select_dtypes(['object'])
    categorical_encoder = dict()

    for categorical_columns_name in categorical_columns_data.columns:
        label_encoder = LabelEncoder()
        categorical_encoder[categorical_columns_name] = label_encoder.fit(
            categorical_columns_data[categorical_columns_name].values)
        priors_data[categorical_columns_name] = label_encoder.fit_transform(
            categorical_columns_data[categorical_columns_name].values)
    priors_data["predictive_accuracy"] = np.array(-1 * top_k_raw_priors.predictive_accuracy.values)
    priors_data.drop(['task_id'], axis=1).to_csv(priors_data_path, index=False)

    categorical_mappping = dict()
    for categorical_columns_name in categorical_columns_data.columns:
        categorical_mappping[categorical_columns_name] = dict()
        classes = categorical_encoder[categorical_columns_name].classes_
        for classe in classes:
            maping = categorical_encoder[categorical_columns_name].transform([classe])
            categorical_mappping[categorical_columns_name][classe] = int(maping[0])

    configuration = {
        "application_name": classifier,
        "categorical_mapping": categorical_mappping,
        "prior_BO_parameter": {
            "classifier": classifier,
            "flow_id": 6970,
            "nb_top_hp": 10,
            "from_task_list": nearest_task,
            "method": "multivariate",
            "cache_dir": "data",
            "seed": 10,
            "path_kde": "/scratch/lmilijao/experiments/cache_kde",
            "study": "OpenML-CC18"
        },
        "optimization_objectives": ["predictive_accuracy"],
        "optimization_method": "prior_guided_optimization",
        "design_of_experiment": {
            "number_of_samples": 10
        },
        "bounding_box_limits": [-1, 0],
        "optimization_iterations": 50,
        # "prior_estimation_file": priors_data_path,
        "estimate_multivariate_priors": True,
        "input_parameters": configspace_to_json(configspace, priors_data)
    }

    json_path = os.path.join(cache_dir, "hypermapper_configuration",
                             f"tid_{task_id}_{classifier}_k_{k}_nb_datasets_{nb_datasets}.json")
    if not os.path.isdir(os.path.join(cache_dir, "hypermapper_configuration")):
        os.mkdir(os.path.join(cache_dir, "hypermapper_configuration"))
    with open(json_path, 'w') as json_file:
        json.dump(configuration, json_file)
    return json_path, categorical_encoder


def set_up_pipeline_for_task(task_id, classifier):
    task = openml.tasks.get_task(task_id)
    datasets = task.get_dataset()
    base, _ = modeltype_to_classifier(classifier)
    X, y, categorical_indicator, attribute_names = datasets.get_data(dataset_format="array",
                                                                     target=datasets.default_target_attribute)
    cat = [index for index, value in enumerate(categorical_indicator) if value == True]
    steps = [('imputation', ConditionalImputer(strategy='median',
                                               fill_empty=0,
                                               categorical_features=cat,
                                               strategy_nominal='most_frequent')),
             ('hotencoding',
              ColumnTransformer(transformers=[('enc', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat)],
                                remainder='passthrough')),
             ('scaling', sklearn.preprocessing.StandardScaler(with_mean=False)),
             ('variencethreshold', sklearn.feature_selection.VarianceThreshold()),
             ('classifier', base)]

    if isinstance(base, RandomForestClassifier) or isinstance(base, AdaBoostClassifier):
        del steps[2]

    # print(steps)

    pipe = Pipeline(steps=steps)
    return pipe
