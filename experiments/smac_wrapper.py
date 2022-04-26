import shutil
from pathlib import Path

import numpy as np
import openml
from ConfigSpace.configuration_space import Configuration
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.model_selection import cross_validate
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from .openml_pimp import set_up_pipeline_for_task

openml.config.logger.propagate = False
openml.datasets.dataset.logger.propagate = False

from smac.callbacks import IncorporateRunResultCallback


openml.config.logger.propagate = False
openml.datasets.dataset.logger.propagate = False


class ResultCallback(IncorporateRunResultCallback):
    def __init__(self, task_id, pipeline, counter):
        self.task_id, self.counter, self.pipeline = task_id, counter, pipeline
        self.results = []

    def __call__(self, smbo, run_info, result, time_left):
        self.results.append({
            "task_id": self.task_id,
            "pipeline": self.pipeline,
            "hp_id": self.counter,
            "hp": run_info.config.get_dictionary(),
            "status": str(result.status),
            "performance": -result.cost
        })
        self.counter += 1


class Runner():

    def __init__(self, pipeline, config_space, seed):
        self.pipeline = pipeline
        self.config_space = config_space
        self.seed = seed

    def generate_black_box_function(self, task_id, n_jobs):
        # task = openml.tasks.get_task(task_id)
        # X, y = task.get_X_and_y()
        # train_idx, test_idx = task.get_train_test_split_indices()
        # X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        def black_box_function(config):
            cfg = config.get_dictionary()
            pipe = set_up_pipeline_for_task(task_id, self.pipeline)
            pipe.set_params(**cfg)
            run = openml.runs.run_model_on_task(pipe, task_id, avoid_duplicate_runs=False,
                                                dataset_format="array", n_jobs=n_jobs)
            return - np.mean(list(run.fold_evaluations['predictive_accuracy'][0].values())), {}

            # scores = cross_validate(pipe, X_train, y_train, cv=5, scoring='balanced_accuracy', return_estimator=True,
            #                         n_jobs=n_jobs)
            # validation_score = np.mean(scores['test_score'])
            # test_score = np.mean(
            #     [balanced_accuracy_score(y_true=y_test, y_pred=estimator.predict(X_test)) for estimator in
            #      scores["estimator"]])
            #
            # return -validation_score, {"validation_score": validation_score, "test_score": test_score}

        return black_box_function

    def clean_dict(self, dictionary):
        c = {"True": True, "False": False}
        return  {k: c[v] if v in c else v for k, v in dictionary.items()}

    def exec(self, task_id, hps, counter):
        init_configs = [Configuration(configuration_space=self.config_space, values=self.clean_dict(hp)) for hp in hps]

        obective_function = self.generate_black_box_function(task_id=task_id, n_jobs=1)

        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": len(init_configs),
                             # max. number of function evaluations; for this example set to a low number
                             "cs": self.config_space,  # configuration space
                             "deterministic": "true",
                             "execdir": "/tmp",
                             "cutoff": 60 * 15,
                             "memory_limit": 5000,
                             "cost_for_crash": 1,
                             "abort_on_first_run_crash": False
                             })

        smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(self.seed),
                        tae_runner=obective_function, initial_configurations=init_configs, initial_design=None)

        cal = ResultCallback(task_id=task_id, pipeline=self.pipeline, counter=counter)
        smac.register_callback(cal)

        smac.optimize()

        path = Path(smac.output_dir)
        parent = path.parent.absolute()
        shutil.rmtree(parent, ignore_errors=True)

        return cal.results
