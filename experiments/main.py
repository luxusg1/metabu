import inspect
import os
import sys

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(currentdir, "externalPythonLib"))

from csv import writer
from dataclasses import dataclass, field
from typing import *

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from optimizer.random_sampling import BestHPNeighborSampling
from optimizer.smac import SMAC_Optimizer
from priors.priors import Distribution
from utils.distance import get_nearest_task
from utils.metafeatures import MetaFeatures
import openmlpimp

import openml

openml.config.cache_directory = os.path.expanduser('~/openml_cache/')
openml.config.logger.propagate = False
openml.config.openml_logger.propagate = False

datasets_has_priors = np.array(
    [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49, 53, 219, 2074, 2079, 3021,
     3022, 3481, 3549,
     3560, 3573, 3902, 3903, 3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971,
     9976, 9977, 9978,
     9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969, 14970, 125920, 125922, 146195, 146800,
     146817, 146819, 146820, 146821, 146824, 167125])


@dataclass
class SearchParams:
    nb_neighbors_datasets: int = 10
    distribution_method: str = "multivariate"
    top_fraction_to_consider: int = 10
    seed: int = 1
    n_jobs: int = 1


@dataclass
class OptimizerParams:
    name: str = "SMBO"
    num_iterations: int = 30


@dataclass
class Classifier_libsvm_svc:
    classifier: str = "libsvm_svc"
    flow_id: int = 7707


@dataclass
class Classifier_adaboot:
    classifier: str = "adaboost"
    flow_id: int = 6970


@dataclass
class Classifier_random_forest:
    classifier: str = "random_forest"
    flow_id: int = 6969


@dataclass
class Classifier_autosklearn:
    classifier: str = "autosklearn"
    flow_id: int = 6969


@dataclass
class Study:
    suite: str = "OpenML-CC18"
    cache: str = os.path.abspath('data/openml/OpenML-CC18')


@dataclass
class OtherParams:
    all_cache_dir: str = os.path.abspath('data')


defaults = [
    {"classifier": "adaboost"},
    {"hydra/launcher": "joblib"}
]


@dataclass
class Config:
    classifier: Any = MISSING
    optimizerParams: OptimizerParams = OptimizerParams()
    searchParams: SearchParams = SearchParams()
    study: Study = Study()
    otherParams: OtherParams = OtherParams()
    defaults: List[Any] = field(default_factory=lambda: defaults)
    path_kde: str = os.path.expanduser("~/pimp_data")
    metafeatures: str = "learned"


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)
cs.store(group="classifier", name="adaboost", node=Classifier_adaboot)
cs.store(group="classifier", name="libsvm_svc", node=Classifier_libsvm_svc)
cs.store(group="classifier", name="random_forest", node=Classifier_random_forest)
cs.store(group="classifier", name="autosklearn", node=Classifier_autosklearn)


@hydra.main(config_name="config")
def main(cfg: Config) -> float:
    run(cfg)


def check_task_if_done(task_id, seed, result_directory, total_iteration):
    result_path = result_directory + '/result_task_id_' + str(task_id) + '_seed_' + str(seed) + '.csv'
    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)

    if os.path.isfile(result_path):
        df = pd.read_csv(result_path, index_col=None, names=['iteration', 'predictive_accuracy', "runtime"])
        if df.shape[0] >= total_iteration:
            return True

    return False


def run(config):
    print("[START] TASK={} seed={}".format(config.task_id, config.searchParams.seed))
    task_id = config.task_id
    print('Loading seed number {} for task {} ...'.format(config.searchParams.seed, task_id))
    if check_task_if_done(task_id, config.searchParams.seed, os.getcwd(), config.optimizerParams.num_iterations):
        print('TASK ID={} (seed={}) already finished.'.format(task_id, config.searchParams.seed))

    list_possible_priors = openmlpimp.utils.priors.get_list_tasks_with_prior(cache_directory=config.path_kde,
                                                                             study_id=config.study.suite,
                                                                             flow_id=config.classifier.flow_id)
    # print("List possible priors:", list_possible_priors)

    if config.metafeatures == "random":
        nearest_task = []  # no need to specify
    elif config.metafeatures in ["autosklearn", "bardenet_2013_boost", "pfahringer_2000_experiment1"]:
        metafeatures = MetaFeatures(cache_directory=config.otherParams.all_cache_dir,
                                    subset=config.metafeatures)
        nearest_task = get_nearest_task(k=50, task_id=task_id,
                                        study=config.study.suite, metafeatures=metafeatures,
                                        normalize=False)
        nearest_task = [dt for dt in nearest_task if dt in list_possible_priors][
                       :config.searchParams.nb_neighbors_datasets]
    elif config.metafeatures == "learned":
        import json
        with open(os.path.join(config.otherParams.all_cache_dir, "nearest", "best",
                               "general_statistical_info-theory_complexity_concept_itemset_clustering_landmarking_model-based",
                               "tid_{}_{}.json".format(task_id, config.classifier.classifier)),
                  'r') as json_file:
            nearest_task = json.load(json_file)
        nearest_task = [dt for dt in nearest_task if dt in datasets_has_priors][
                       :config.searchParams.nb_neighbors_datasets]

    distribution = Distribution(nb_top_hp=config.searchParams.top_fraction_to_consider,
                                metafeatures=config.metafeatures,
                                from_task_list=nearest_task,
                                classifier=config.classifier.classifier,
                                study=config.study.suite,
                                path_kde=config.path_kde,
                                cache_dir=config.otherParams.all_cache_dir,
                                seed=config.searchParams.seed)

    if config.optimizerParams.name == "random_sampling":
        optimizer = BestHPNeighborSampling(hp_distribution=distribution, classifier=config.classifier.classifier,
                                           seed=config.searchParams.seed,
                                           nb_iterations=config.optimizerParams.num_iterations,
                                           flow_id=config.classifier.flow_id,
                                           n_jobs=config.searchParams.n_jobs)

    elif config.optimizerParams.name == "smac":
        optimizer = SMAC_Optimizer(hp_distribution=distribution, classifier=config.classifier.classifier,
                                   seed=config.searchParams.seed,
                                   nb_iterations=config.optimizerParams.num_iterations,
                                   flow_id=config.classifier.flow_id,
                                   n_jobs=config.searchParams.n_jobs, nb_init=10)
    else:
        raise Exception('optimizer {} is not implemented'.format(config.optimizerParams.name))

    optimizer.run(task_id, n_jobs=config.searchParams.n_jobs)
    all_iteration_measure = optimizer.collection.iloc[-config.optimizerParams.num_iterations:][
        ['predictive_accuracy', 'runtime']]

    file = os.getcwd() + '/result_task_id_' + str(task_id) + '_seed_' + str(config.searchParams.seed) + '.csv'

    if all_iteration_measure.shape[0] > 0:
        cols_name = ['iteration', 'predictive_accuracy', 'runtime']
        df = pd.DataFrame(columns=cols_name)
        df.to_csv(file, index=False)

    iteration_number = 1
    for _, row in all_iteration_measure.iterrows():
        List = [iteration_number, row['predictive_accuracy'], row['runtime']]
        with open(file, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
        iteration_number += 1

    print('[END] TASK ID={} (seed={}) finished.'.format(task_id, config.searchParams.seed))


if __name__ == "__main__":
    print(main())
