import collections
import scipy
import copy


def rank_dict(dictionary, reverse=False):
    dictionary = copy.copy(dictionary)
    if reverse:
        for key in dictionary.keys():
            dictionary[key] = 1 - dictionary[key]
    sortdict = collections.OrderedDict(sorted(dictionary.items()))
    ranks = scipy.stats.rankdata(list(sortdict.values()))
    result = {}
    for idx, (key, value) in enumerate(sortdict.items()):
        result[key] = ranks[idx]
    return result


def sum_dict_values(a, b, allow_subsets=False):
    result = {}
    a_total = sum(a.values())
    b_total = sum(b.values())
    a_min_b = set(a.keys()) - set(b.keys())
    b_min_a = set(b.keys()) - set(a.keys())
    if len(b_min_a) > 0:
        raise ValueError('dict b got illegal keys: %s' %str(b_min_a))
    if not allow_subsets and len(a_min_b):
        raise ValueError('keys not the same')
    for idx in a.keys():
        if idx in b:
            result[idx] = a[idx] + b[idx]
        else:
            result[idx] = a[idx]
    if sum(result.values()) != a_total + b_total:
        raise ValueError()
    return result


def divide_dict_values(d, denominator):
    result = {}
    for idx in d.keys():
        result[idx] = d[idx] / denominator
    return result