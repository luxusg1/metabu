import sys
sys.path.append('..')

import shutil
from pathlib import Path

import numpy as np
import openml
import pandas as pd
import sklearn
from ConfigSpace.configuration_space import Configuration
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from openmlpimp.utils import modeltype_to_classifier
from openmlstudy14.preprocessing import ConditionalImputer
from utils.hyperparameters import get_configspace

openml.config.logger.propagate = False
openml.datasets.dataset.logger.propagate = False


def set_up_pipeline_for_task(task_id, classifier):
    task = openml.tasks.get_task(task_id)
    datasets = task.get_dataset()
    base, _ = modeltype_to_classifier(classifier)
    _, _, categorical_indicator, attribute_names = datasets.get_data(dataset_format="array",
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

    pipe = Pipeline(steps=steps)
    return pipe


class SMAC_Optimizer():

    def __init__(self, hp_distribution, classifier, flow_id, seed, nb_iterations, n_jobs, nb_init):

        self.distribution = hp_distribution
        self.classifier = classifier
        self.config_space = get_configspace(classifier, 42)
        self.flow_id = flow_id
        self.seed = seed
        self.from_task_list = []
        self.nb_init = nb_init

        self.scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                  "runcount-limit": nb_iterations,
                                  # max. number of function evaluations; for this example set to a low number
                                  "cs": self.config_space,  # configuration space
                                  "deterministic": "true",
                                  "execdir": "/tmp",
                                  "cutoff": 60 * 15,
                                  "memory_limit": 5000 * n_jobs,
                                  "cost_for_crash": 1,
                                  "abort_on_first_run_crash": False
                                  })

    def add_new_collection(self, hp, predictive_accuracy, runtime):
        d = dict()
        for name in self.config_space.get_hyperparameter_names():
            if name in hp:
                d[name] = hp[name]
        d["predictive_accuracy"] = predictive_accuracy
        d["runtime"] = runtime
        return pd.DataFrame([d])

    def generate_black_box_function(self, task_id, n_jobs):
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        train_idx, test_idx = task.get_train_test_split_indices()
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        def black_box_function(config):
            cfg = config.get_dictionary()
            pipe = set_up_pipeline_for_task(task_id, self.classifier)
            pipe.set_params(**cfg)
            # run = openml.runs.run_model_on_task(pipe, task, avoid_duplicate_runs=False,
            #                                     dataset_format="array", n_jobs=n_jobs)
            # predictive_accuracy = np.mean(list(run.fold_evaluations['predictive_accuracy'][0].values()))

            scores = cross_validate(pipe, X_train, y_train, cv=5, scoring='balanced_accuracy', return_estimator=True,
                                    n_jobs=n_jobs)
            validation_score = np.mean(scores['test_score'])
            test_score = np.mean(
                [balanced_accuracy_score(y_true=y_test, y_pred=estimator.predict(X_test)) for estimator in
                 scores["estimator"]])

            return -validation_score, {"validation_score": validation_score, "test_score": test_score}

        def run_on_autosklearn(config):
            classifier = SimpleClassificationPipeline(config=config, random_state=self.seed)

            scores = cross_validate(classifier, X_train, y_train, cv=5, scoring='balanced_accuracy',
                                    return_estimator=True,
                                    n_jobs=n_jobs)
            validation_score = np.mean(scores['test_score'])
            test_score = np.mean(
                [balanced_accuracy_score(y_true=y_test, y_pred=estimator.predict(X_test)) for estimator in
                 scores["estimator"]])

            return -validation_score, {"validation_score": validation_score, "test_score": test_score}

        return black_box_function if self.classifier in ["random_forest", "adaboost",
                                                         "libsvm_svc"] else run_on_autosklearn

    def _init_collection(self):
        columns_name = []
        for name in self.config_space.get_hyperparameter_names():
            columns_name.append(name)
        columns_name.append("predictive_accuracy")
        columns_name.append("runtime")

        self.collection = pd.DataFrame(columns=columns_name)

    def run(self, task_id, n_jobs=None):
        self._init_collection()

        init_configs = [Configuration(configuration_space=self.config_space, values=hp) for hp in
                        self.distribution.sample(nb_samples=self.nb_init)]
        print("list init configurations", init_configs)

        obective_function = self.generate_black_box_function(task_id=task_id, n_jobs=n_jobs)
        smac = SMAC4HPO(scenario=self.scenario, rng=np.random.RandomState(self.seed),
                        tae_runner=obective_function, initial_configurations=init_configs, initial_design=None)
        smac.optimize()
        results = list(smac.runhistory.data.values())
        list_configs = [_.get_dictionary() for _ in smac.runhistory.get_all_configs()]

        current_validation_score, current_test_score = 0, 0

        for hp, run in zip(list_configs, results):
            if run.status == run.status.SUCCESS and current_validation_score < run.additional_info["validation_score"]:
                current_validation_score = run.additional_info["validation_score"]
                current_test_score = run.additional_info["test_score"]
            new_collection = self.add_new_collection(hp, current_test_score, run.time)
            self.collection = self.collection.append(new_collection)
        path = Path(smac.output_dir)
        parent = path.parent.absolute()
        shutil.rmtree(parent, ignore_errors=True)
