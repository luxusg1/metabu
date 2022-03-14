'''import openml
import numpy as np


meta_features_auto_sklearn = [
'NumberOfInstances',
'NumberOfClasses',
'NumberOfFeatures',
'NumberOfMissingValues',
'NumberOfInstancesWithMissingValues',
'PercentageOfInstancesWithMissingValues',
'PercentageOfMissingValues',
'NumberOfNumericFeatures',
'NumberOfSymbolicFeatures',
'PercentageOfBinaryFeatures',
'PercentageOfNumericFeatures',
'NumberOfBinaryFeatures',
'PercentageOfSymbolicFeatures',
'ClassEntropy'
]

def get_data_features(task_id):
    data_qualities = openml.tasks.get_task(task_id).get_dataset().qualities
    result = np.array(list(map(data_qualities.get, meta_features_auto_sklearn)))
    return result
'''

import logging
import os
from io import StringIO
from unittest import TestCase

import autosklearn.metalearning.metafeatures.metafeatures as meta_features
import numpy as np
import openml
import pandas as pd
from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor


import sys
sys.path.append('..')

from utils.utils import get_suite, normalize_dataframe

subsets = dict()

subsets["autosklearn"] = set(["ClassEntropy",
                              "SkewnessSTD",
                              "SkewnessMean",
                              "SkewnessMax",
                              "SkewnessMin",
                              "KurtosisSTD",
                              "KurtosisMean",
                              "KurtosisMax",
                              "KurtosisMin",
                              "SymbolsSum",
                              "SymbolsSTD",
                              "SymbolsMean",
                              "SymbolsMax",
                              "SymbolsMin",
                              "ClassProbabilitySTD",
                              "ClassProbabilityMean",
                              "ClassProbabilityMax",
                              "ClassProbabilityMin",
                              "InverseDatasetRatio",
                              "DatasetRatio",
                              "RatioNominalToNumerical",
                              "RatioNumericalToNominal",
                              "NumberOfCategoricalFeatures",
                              "NumberOfNumericFeatures",
                              "NumberOfMissingValues",
                              "NumberOfFeaturesWithMissingValues",
                              "NumberOfInstancesWithMissingValues",
                              "NumberOfFeatures",
                              "NumberOfClasses",
                              "NumberOfInstances",
                              "LogInverseDatasetRatio",
                              "LogDatasetRatio",
                              "PercentageOfMissingValues",
                              "PercentageOfFeaturesWithMissingValues",
                              "PercentageOfInstancesWithMissingValues",
                              "LogNumberOfFeatures",
                              "LogNumberOfInstances"])

subsets["npy"] = set(["LandmarkLDA",
                      "LandmarkNaiveBayes",
                      "LandmarkDecisionTree",
                      "LandmarkDecisionNodeLearner",
                      "LandmarkRandomNodeLearner",
                      "LandmarkWorstNodeLearner",
                      "Landmark1NN",
                      "PCAFractionOfComponentsFor95PercentVariance",
                      "PCAKurtosisFirstPC",
                      "PCASkewnessFirstPC",
                      "Skewnesses",
                      "SkewnessMin",
                      "SkewnessMax",
                      "SkewnessMean",
                      "SkewnessSTD",
                      "Kurtosisses",
                      "KurtosisMin",
                      "KurtosisMax",
                      "KurtosisMean",
                      "KurtosisSTD"])

# Metafeatures used by Pfahringer et al. (2000) in the first experiment
subsets["pfahringer_2000_experiment1"] = set(['NumberOfFeatures',
                                              'NumberOfNumericFeatures',
                                              'NumberOfCategoricalFeatures',
                                              'NumberOfClasses',
                                              'ClassProbabilityMax',
                                              "LandmarkLDA",
                                              "LandmarkNaiveBayes",
                                              "LandmarkDecisionTree"])

# Metafeatures used by Pfahringer et al. (2000) in the second experiment
# worst node learner not implemented yet

'''subsets["pfahringer_2000_experiment2"] = set(["LandmarkDecisionNodeLearner",
                                   "LandmarkRandomNodeLearner",
                                   "LandmarkWorstNodeLearner",
                                   "Landmark1NN"])
                                   '''

# Metafeatures used by Yogatama and Mann (2014)
subsets["yogotama_2014"] = set(['LogNumberOfFeatures',
                                'LogNumberOfInstances',
                                'NumberOfClasses'])

# Metafeatures used by Bardenet et al. (2013) for the AdaBoost.MH experiment
subsets["bardenet_2013_boost"] = set(['NumberOfClasses',
                                      'LogNumberOfFeatures',
                                      'LogInverseDatasetRatio',
                                      'PCAFractionOfComponentsFor95PercentVariance'])

# Metafeatures used by Bardenet et al. (2013) for the Neural Net experiment
subsets["bardenet_2013_nn"] = set(['NumberOfClasses',
                                   'LogNumberOfFeatures',
                                   'LogInverseDatasetRatio',
                                   "PCAKurtosisFirstPC",
                                   "PCASkewnessFirstPC"])

subsets["test"] = set(["PCASkewnessFirstPC"])


class MetaFeatures(TestCase):
    _multiprocess_can_split_ = True

    def __init__(self, cache_directory=os.path.expanduser("~/experiments"), subset="all"):
        super().__init__()
        self.cache_directory = cache_directory
        self.subset = subset
        self.meta_features_cache_dir = os.path.join(cache_directory, 'metafeatures')

    def setUp(self, task_id):
        task = openml.tasks.get_task(task_id)
        datasets = task.get_dataset()
        self.nominal = datasets.get_features_by_type('nominal', [task.target_name])
        self.categorical = [True if i in self.nominal else False for i in range(len(datasets.get_data()[0].dtypes) - 1)]
        X, y = task.get_X_and_y()

        DPP = DataPreprocessor(categorical_features=self.categorical)
        X_transformed = DPP.fit_transform(X)

        # Transform the array which indicates the categorical metafeatures
        number_numerical = np.sum(~np.array(self.categorical))
        categorical_transformed = [True] * (X_transformed.shape[1] -
                                            number_numerical) + \
                                  [False] * number_numerical
        self.categorical_transformed = categorical_transformed

        self.X = X
        self.X_transformed = X_transformed
        self.y = y
        self.mf = meta_features.metafeatures
        self.helpers = meta_features.helper_functions

        # Create a logger for testing
        self.logger = logging.getLogger()

        # Precompute some helper functions
        self.helpers.set_value(
            "PCA", self.helpers["PCA"](self.X_transformed, self.y, self.logger),
        )
        self.helpers.set_value(
            "MissingValues",
            self.helpers["MissingValues"](self.X, self.y, self.logger, self.categorical),
        )
        self.helpers.set_value(
            "NumSymbols",
            self.helpers["NumSymbols"](self.X, self.y, self.logger, self.categorical),
        )
        self.helpers.set_value(
            "ClassOccurences",
            self.helpers["ClassOccurences"](self.X, self.y, self.logger),
        )
        self.helpers.set_value(
            "Skewnesses",
            self.helpers["Skewnesses"](self.X_transformed, self.y,
                                       self.logger, self.categorical_transformed),
        )
        self.helpers.set_value(
            "Kurtosisses",
            self.helpers["Kurtosisses"](self.X_transformed, self.y,
                                        self.logger, self.categorical_transformed),
        )

    def calculate(self):
        mf = meta_features.calculate_all_metafeatures(self.X, self.y, self.categorical, "test", logger=self.logger)
        # print(task_metafeatures.metafeature_values)
        sio = StringIO()
        mf.dump(sio)
        dict_metafeatures = dict()

        for metafeatures_name, metafeatures_config in self.mf.values.items():
            dict_metafeatures[metafeatures_name] = metafeatures_config.value
        print(len(dict_metafeatures))
        return dict_metafeatures

    def get_by_task_list(self, task_list):
        path_metafeatures_study = os.path.join(self.meta_features_cache_dir, 'metafeatures_scot_task.csv')
        metafeatures_by_study = []
        if not os.path.isfile(path_metafeatures_study):
            for task in task_list:
                print("compute metafeatures for task " + str(task))
                self.setUp(task)
                task_metafeatures = self.calculate()
                task_metafeatures['task_id'] = task
                metafeatures_by_study.append(task_metafeatures)
            df_metafeatures_by_study = pd.DataFrame(metafeatures_by_study)
            df_metafeatures_by_study.to_csv(path_metafeatures_study, index=False)
        else:
            df_metafeatures_by_study = pd.read_csv(path_metafeatures_study)
        return self.__filters_by_subsets(df_metafeatures_by_study)

    def get_all_by_study(self, study):
        if not os.path.isdir(self.meta_features_cache_dir):
            os.makedirs(self.meta_features_cache_dir)

        path_metafeatures_study = os.path.join(self.meta_features_cache_dir, 'metafeatures_' + study + '.csv')
        study_cache_dir = self.cache_directory

        metafeatures_by_study = []
        if not os.path.isfile(path_metafeatures_study):
            for task in get_suite(study_cache_dir, study=study).tasks:
                print("compute metafeatures for task " + str(task))
                self.setUp(task)
                task_metafeatures = self.calculate()
                # print(task_metafeatures.keys())
                task_metafeatures['task_id'] = task
                metafeatures_by_study.append(task_metafeatures)
            df_metafeatures_by_study = pd.DataFrame(metafeatures_by_study)
            df_metafeatures_by_study.to_csv(path_metafeatures_study, index=False)
        else:
            df_metafeatures_by_study = pd.read_csv(path_metafeatures_study)
        return self.__filters_by_subsets(df_metafeatures_by_study)

    def __filters_by_subsets(self, metafeatures):
        if self.subset == 'all':
            return metafeatures
        subset_filter = list(subsets[self.subset])
        subset_filter.append('task_id')
        return metafeatures[set(subset_filter)]

    def get_by_task(self, task_id, study, normalize=True):
        metafeatures_by_study = self.get_all_by_study(study)
        if normalize:
            index = metafeatures_by_study[metafeatures_by_study.task_id == task_id].index[0]
            metafeatures_by_study = normalize_dataframe(metafeatures_by_study, holdout=['task_id'], index=index)

        task_metafeatures = metafeatures_by_study[metafeatures_by_study['task_id'] == task_id]
        return task_metafeatures
