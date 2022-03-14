import os
import random
import numpy as np
from ConfigSpace.configuration_space import Configuration

import sys
sys.path.append('..')
from utils.hyperparameters import get_topk_priors, get_configspace


class Distribution(object):
    def __init__(self, nb_top_hp, from_task_list, metafeatures, classifier,
                 cache_dir, seed, path_kde, study="OpenML-CC18"):
        self.classifier = classifier
        self.cache_dir = cache_dir

        self.fixed_parameters = None
        self.configspace = get_configspace(self.classifier, seed)
        self.seed = seed
        self.study = study
        self.from_task_list = from_task_list
        self.path_cache_kde = path_kde
        path_prior = os.path.join(cache_dir, "data", f"{classifier}_best_all.arff")
        if metafeatures != "random":
            self.list_configurations, self.sample_weights = get_topk_priors(cache_dir=path_prior,
                                                                                classifier=classifier,
                                                                                list_tid=from_task_list,
                                                                                k=nb_top_hp,
                                                                                seed=seed,
                                                                                with_weight=True)
        self.metafeatures = metafeatures

    def sample(self, nb_samples, **kwargs):
        if self.metafeatures == "random":
            return self.sample_uniform(nb_samples=nb_samples)
        return self.sample_from_distribution(nb_samples=nb_samples, **kwargs)


    def sample_uniform(self, nb_samples: int):
        samples = self.configspace.sample_configuration(nb_samples)
        if isinstance(samples, Configuration): return [samples]
        return samples

    def sample_from_distribution(self, nb_samples: int):
        return random.choices(self.list_configurations, k=nb_samples, weights=self.sample_weights)

