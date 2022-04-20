
import os
import json
import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, OrdinalHyperparameter, Constant
from ConfigSpace.configuration_space import Configuration
from openmlpimp.configspaces import adaboost, random_forest, libsvm_svc
from scipy.io import arff
from ConfigSpace.read_and_write import json as json_config_space
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm

datasets_has_priors = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49, 53, 219, 2074, 2079, 3021,
                       3022, 3481, 3549,
                       3560, 3573, 3902, 3903, 3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971,
                       9976, 9977, 9978,
                       9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 125920, 125922, 146195, 146800,
                       146817, 146819,
                       146820, 146821, 146824, 167125]


def get_configspace(classifier, seed):
    if classifier == "adaboost":
        return adaboost.get_hyperparameter_search_space(seed)
    elif classifier == "random_forest":
        return random_forest.get_hyperparameter_search_space(seed)
    elif classifier == "libsvm_svc":
        return libsvm_svc.get_hyperparameter_search_space(seed)
    elif classifier == "autosklearn":
        path_margaret = "/home/tau/lmilijao/priors_BO/"
        path_jean_zay = "/home/louisot/priors_BO/"  # "/linkhome/rech/genini01/uvp29is/Code/metabu"
        path = path_margaret if os.path.exists(path_margaret) else path_jean_zay
        data_dir = os.path.join(path, "data/data")

        with open(os.path.join(data_dir, 'config_space_autosklearn.json'), 'r') as f:
            jason_string = f.read()
            return json_config_space.read(jason_string)
    elif classifier == 'pmf':
        path_margaret = "/home/tau/lmilijao/priors_BO/"
        path_jean_zay = "/home/louisot/priors_BO/"  # "/linkhome/rech/genini01/uvp29is/Code/metabu"
        path = path_margaret if os.path.exists(path_margaret) else path_jean_zay
        data_dir = os.path.join(path, "data/data")

        with open(os.path.join(data_dir, 'config_space_PMF.json'), 'r') as f:
            jason_string = f.read()
            return json_config_space.read(jason_string)
    raise Exception(f"Classifier {classifier} not recognized.")


def get_hp_types(configspace):
    list_hps = configspace.get_hyperparameters()
    list_hp_numerical, list_hp_categorical, list_hp_constant = [], [], []

    for idx, hp in enumerate(list_hps):
        if isinstance(hp, CategoricalHyperparameter):
            list_hp_categorical.append(idx)
        elif isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter, OrdinalHyperparameter)):
            list_hp_numerical.append(idx)
        elif isinstance(hp, Constant):
            list_hp_constant.append(idx)

    assert (len(list_hp_constant) + len(list_hp_numerical) + len(list_hp_categorical)) == len(
        configspace.sample_configuration().get_array())
    return list_hp_categorical, list_hp_numerical, list_hp_constant


def get_raw_priors(cache_dir: str, classifier: str):
    if classifier != "autosklearn":
        df_hp = pd.DataFrame(arff.loadarff(cache_dir)[0])
        if classifier == "random_forest":
            for columns_name in ['classifier__bootstrap', 'classifier__criterion', 'imputation__strategy']:
                df_hp[columns_name] = df_hp[columns_name].str.decode('utf-8')
            df_hp = df_hp.astype({"classifier__max_features": float,
                                  "classifier__min_samples_split": int,
                                  "classifier__min_samples_leaf": int})
        if classifier == "adaboost":
            for columns_name in ['classifier__algorithm', 'imputation__strategy']:
                df_hp[columns_name] = df_hp[columns_name].str.decode('utf-8')
            df_hp.classifier__algorithm = np.where((df_hp.classifier__algorithm == 'SAMME_R'),
                                                   'SAMME.R', df_hp.classifier__algorithm)
            df_hp = df_hp.astype({"classifier__base_estimator__max_depth": int,
                                  "classifier__learning_rate": float,
                                  "classifier__n_estimators": int})

        if classifier == "libsvm_svc":
            for columns_name in ['classifier__kernel', 'imputation__strategy']:
                df_hp[columns_name] = df_hp[columns_name].str.decode('utf-8')

            df_hp = df_hp.astype({"classifier__C": float,
                                  "classifier__degree": int,
                                  "classifier__gamma": float,
                                  "classifier__coef0": float,
                                  "classifier__tol": float,
                                  "classifier__max_iter": int,
                                  "classifier__shrinking": bool})

        if df_hp.task_id.dtype == 'object':
            df_hp.task_id = df_hp.task_id.str.decode('utf-8')
            df_hp.task_id = df_hp.task_id.astype('int64')

        return df_hp
    else:
        with open(cache_dir) as json_file:
            list_configs = json.load(json_file)
        return list_configs


def get_significative_best(data, k):
    if 'predictive_performance' in data.columns:
        scores = data.predictive_performance.values
    else:
        scores = data.predictive_accuracy.values
    threshold = np.percentile(scores, q=100 - k)
    idx = np.where(scores >= threshold)[0]
    return data.iloc[idx]


def get_weight(iteration):
    beta = 5
    return np.exp(-iteration / 5)


def to_complete_config(config, classifier):
    configspace = get_configspace(classifier, 1)
    tmp = np.array([])
    for hp in configspace.get_hyperparameters():
        if hp.name in config.keys():
            if isinstance(hp, CategoricalHyperparameter):
                tmp = np.append(tmp, [hp.choices.index(config[hp.name])])
            else:
                if isinstance(hp, Constant):
                    tmp = np.append(tmp, [0])
                elif isinstance(hp, UniformFloatHyperparameter) or isinstance(hp, UniformIntegerHyperparameter):
                    if hp.lower <= config[hp.name] <= hp.upper:
                        tmp = np.append(tmp, [config[hp.name]])
                    else:
                        print(hp.name, hp.lower, config[hp.name], hp.upper)
                        raise Exception('problem')
                else:
                    tmp = np.append(tmp, [config[hp.name]])
        else:
            if isinstance(hp, CategoricalHyperparameter):
                tmp = np.append(tmp, [hp.choices.index(hp.default_value)])
            else:
                if isinstance(hp, Constant):
                    tmp = np.append(tmp, [0])
                elif isinstance(hp, UniformFloatHyperparameter) or isinstance(hp, UniformIntegerHyperparameter):
                    tmp = np.append(tmp, [hp.default_value])
                else:
                    tmp = np.append(tmp, [hp.default_value])

    return tmp


def get_path():
    margaret_path = '/home/tau/lmilijao/priors_BO/'
    local_path = '/home/louisot/priors_BO/'
    if os.path.isdir(margaret_path):
        return margaret_path
    else:
        return local_path


def get_top_k_label_encoded_raw_priors_old(cache_dir: str, classifier: str, list_tid: list, k: int,
                                           method: str = 'all'):
    df_hp = get_raw_priors(cache_dir=cache_dir, classifier=classifier)
    cs = get_configspace(classifier, 1)
    topk_dataset = []
    have_file_data = False
    if classifier == 'autosklearn':
        file_path = os.path.join(get_path(), 'data/data/fanova_data/autosklearn/total.csv')
        if os.path.isfile(file_path):
            df_hp = pd.read_csv(file_path)
            have_file_data = True
        else:
            df_hp = pd.DataFrame(df_hp)
        perf_measure_name = 'predictive_performance'
    else:
        perf_measure_name = 'predictive_accuracy'

    # get the top k

    if method == 'all':
        df_topk = df_hp
    else:
        for tid in list_tid:
            # tmp = df_hp[df_hp.task_id == tid].sort_values(by=[perf_measure_name], ascending=False)[:k]
            # if tmp.shape[0] != k: raise Exception(
            #    "Do not have enough sampling on dataset tid:{}. Only {} samples".format(tid, tmp.shape[0]))
            tmp = get_significative_best(df_hp[df_hp.task_id == tid].copy(), k)
            if tmp.shape[0] == 0: raise Exception(
                "Do not have enough sampling on dataset tid:{}. Only {} samples".format(tid, tmp.shape[0]))
            topk_dataset.append(tmp)
        df_topk = pd.concat(topk_dataset, axis=0)

    # transform each configuration of top to complete autosklearn reasearch space

    if classifier == 'autosklearn':
        if not have_file_data:
            column = list(cs.get_hyperparameter_names())
            column.extend(['task_id', 'predictive_accuracy'])
            topk_data = pd.DataFrame(columns=column)
            topk_dict = df_topk.to_dict('records')
            for perf in tqdm(topk_dict):
                cfg = perf['configuration']
                array_cfg = to_complete_config(cfg, 'autosklearn')
                array_cfg = np.append(array_cfg, [perf['task_id'], perf['predictive_performance']])
                topk_data = topk_data.append(pd.DataFrame(columns=column, data=array_cfg.reshape((1, -1))))
            df_topk = topk_data
    elif classifier == 'libsvm_svc':
        column = list(cs.get_hyperparameter_names())
        column.extend(['task_id', 'predictive_accuracy'])
        topk_data = pd.DataFrame(columns=column)
        topk_dict = df_topk.to_dict('records')
        for perf in topk_dict:
            t_id = perf['task_id']
            predictive_accuracy = perf['predictive_accuracy']
            perf.pop('predictive_accuracy')
            perf.pop('task_id')
            array_cfg = to_complete_config(perf, 'libsvm_svc')
            array_cfg = np.append(array_cfg, [t_id, predictive_accuracy])
            topk_data = topk_data.append(pd.DataFrame(columns=column, data=array_cfg.reshape((1, -1))))
        df_topk = topk_data

    else:
        for hp in cs.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                mapping = dict()
                for item in hp.choices:
                    mapping[item] = hp.choices.index(item)
                df_topk[hp.name] = df_topk[hp.name].replace(mapping)
    return df_topk


def get_top_k_raw_priors(cache_dir: str, classifier: str, list_tid: list, k: int, with_weight: bool = False):
    df_hp = get_raw_priors(cache_dir=cache_dir, classifier=classifier)
    if classifier != "autosklearn":
        topk_dataset = []
        weights = []

        for it, tid in enumerate(list_tid):
            w = get_weight(it)
            tmp = get_significative_best(df_hp[df_hp.task_id == tid].copy(), k)

            if tmp.shape[0] == 0: raise Exception(
                "Do not have enough sampling on dataset tid:{}. Only {} samples".format(tid, tmp.shape[0]))
            topk_dataset.append(tmp)
            weights.extend([w / len(tmp) for _ in range(len(tmp))])
        if with_weight:
            return pd.concat(topk_dataset, axis=0), np.array(weights)
        return pd.concat(topk_dataset, axis=0)
    else:
        index_list = [{"task_id": val["task_id"], "predictive_accuracy": val["predictive_performance"]} for val in
                      df_hp]
        df_index = pd.DataFrame(index_list)

        index_to_store = []
        weights = []

        for it, tid in enumerate(list_tid):
            w = get_weight(it)
            tmp = get_significative_best(df_index[df_index.task_id == tid].copy(), k)
            index_to_store.extend(tmp.index.tolist())
            weights.extend([w / len(tmp) for _ in range(len(tmp))])
        index_to_store = set(index_to_store)
        top_k_configs = [_ for i, _ in enumerate(df_hp) if i in index_to_store]
        if with_weight:
            return top_k_configs, np.array(weights)
        return top_k_configs


def preprocessed_df_priors(df, configspace):
    if isinstance(df, pd.DataFrame):
        hp_json = df.drop(["task_id", "predictive_accuracy"], axis=1).to_dict('records')
    elif isinstance(df, list):
        hp_json = [val["configuration"] for val in df]
    list_configuration = []
    for hp in hp_json:
        cfg = Configuration(configuration_space=configspace, values=hp, allow_inactive_with_values=True)
        list_configuration.append(cfg.get_array())
    return np.nan_to_num(np.array(list_configuration))


def _get_preprocessing_pipeline(config_space):
    list_hp_categorical, list_hp_numerical, list_hp_constant = get_hp_types(configspace=config_space)
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='error', sparse=False)
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, list_hp_numerical),
            ('cat', categorical_transformer, list_hp_categorical)])


def get_preprocessed_priors(cache_dir: str, classifier: str, seed: int):
    df_hp = get_raw_priors(cache_dir=cache_dir, classifier=classifier)
    cs = get_configspace(classifier=classifier, seed=seed)
    df_hp_preprocess = preprocessed_df_priors(df=df_hp, configspace=cs)

    preprocessor = _get_preprocessing_pipeline(config_space=cs)
    # pipeline = Pipeline(steps=[('preprocess', preprocessor), ('dim_reduction', PCA(n_components=8))])

    if classifier == "libsvm_svc":
        return df_hp_preprocess, preprocessor.fit(
            np.nan_to_num([_.get_array() for _ in cs.sample_configuration(10000)]))
    return df_hp_preprocess, preprocessor.fit(df_hp_preprocess)


def get_preprocessed_topk_priors(cache_dir: str, classifier: str, list_tid: list, k: int, seed: int,
                                 with_weight: bool = False):
    df_hp, weights = get_top_k_raw_priors(cache_dir=cache_dir,
                                          classifier=classifier,
                                          list_tid=datasets_has_priors if list_tid is None else list_tid,
                                          k=k, with_weight=with_weight)
    if with_weight:
        return preprocessed_df_priors(df=df_hp,
                                      configspace=get_configspace(classifier=classifier, seed=seed)), weights
    else:
        return preprocessed_df_priors(df=df_hp,
                                      configspace=get_configspace(classifier=classifier, seed=seed))


def get_topk_priors(cache_dir: str, classifier: str, list_tid: list, k: int, seed: int,
                    with_weight: bool = False):
    res = get_top_k_raw_priors(cache_dir=cache_dir,
                               classifier=classifier,
                               list_tid=datasets_has_priors if list_tid is None else list_tid,
                               k=k, with_weight=with_weight)
    if with_weight:
        df_hp, weights = res
    else:
        df_hp = res
    cs = get_configspace(classifier=classifier, seed=seed)
    if isinstance(df_hp, pd.DataFrame):
        hp_json = df_hp.drop(["task_id", "predictive_accuracy"], axis=1).to_dict('records')
    elif isinstance(df_hp, list):
        hp_json = [val["configuration"] for val in df_hp]

    list_configuration = [Configuration(configuration_space=cs, values=hp, allow_inactive_with_values=True) for hp in
                          hp_json]

    if with_weight:
        return list_configuration, weights
    else:
        return list_configuration
