import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def intrinsic_estimator(matrix_distance):
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


def get_significative_best(data, k, ranking_column_name):
    if 'predictive_performance' in data.columns:
        scores = data.predictive_performance.values
    else:
        scores = data[ranking_column_name].values

    threshold = np.percentile(scores, q=100-k)
    idx = np.where(scores >= threshold)[0]
    return data.iloc[idx]


def get_top_k_target(df, list_tid, k, ranking_column_name):
    topk_dataset = []
    weights = []
    for it, tid in enumerate(list_tid):
        tmp = get_significative_best(df[df.task_id == tid].copy(), k, ranking_column_name)
        if tmp.shape[0] == 0: raise Exception(
            "Do not have enough sampling on dataset tid:{}. Only {} samples".format(tid, tmp.shape[0]))
        topk_dataset.append(tmp)
    return pd.concat(topk_dataset, axis=0)


