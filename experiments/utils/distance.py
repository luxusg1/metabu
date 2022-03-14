from numpy.linalg import norm
import openml

import sys

sys.path.append('..')

from utils.metafeatures import MetaFeatures
from utils.utils import get_suite
import operator
import os


# def compute_distance(x1,x2):
def get_nearest_task(k=10, task_id=3, study="OpenML-CC18", metafeatures=MetaFeatures(),
                     cache_dir=os.path.expanduser("~/experiments"), normalize=True):
    task_metafeatures1 = metafeatures.get_by_task(study=study, task_id=task_id, normalize=normalize)
    distance_dict = dict()

    for task in get_suite(cache_dir=cache_dir, study=study).tasks:
        if task_id != task:
            task_metafeatures2 = metafeatures.get_by_task(study=study, task_id=task, normalize=normalize)
            distance_dict[task] = norm(task_metafeatures1.values - task_metafeatures2.values)

    nearest_task = dict(sorted(distance_dict.items(), key=operator.itemgetter(1))[:k]).keys()
    return nearest_task
