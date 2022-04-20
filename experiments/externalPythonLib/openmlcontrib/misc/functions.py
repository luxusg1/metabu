import openml
import os
import pandas as pd
import typing


def filter_listing(listing, property_name, allowed_values, dict_representation=True):
    """
    Removes items from the result of a listing fn if a property
    does not comply with a given value. 

    Parameters
    ----------
    setupid_setup : dict of dicts or objects, representing
        openml objects as obtained from an openml listing fn

    property_name : str
        the name of the property which values should be restricted

    allowed_values : list
        list of allowed values

    dict_representation : bool
        wether the individual items are represented as dicts
        or objects

    Returns
    -------
    model : dict of dicts or objects
        a dict, with the objects that did not comply removed
    """
    allowed = dict()
    if not isinstance(allowed_values, list):
        raise ValueError('allowed values should be a list')

    for id, object in listing.items():
        if dict_representation:
            if property_name not in object:
                raise ValueError('dict does not have property: %s' %property_name)

            if object[property_name] in allowed_values:
                allowed[id] = object
        else:
            if not hasattr(object, property_name):
                raise ValueError('dict does not have property: %s' % property_name)

            if getattr(object, property_name) in allowed_values:
                allowed[id] = object

    return allowed


def _traverse_run_folders(folder: str, traversed_directories: typing.List) \
        -> typing.List[typing.Tuple[typing.List[str], openml.runs.OpenMLRun]]:
    folder_content = os.listdir(folder)
    if 'description.xml' in folder_content and 'predictions.arff' in folder_content:
        run = openml.runs.OpenMLRun.from_filesystem(folder, expect_model=False)
        return [(traversed_directories, run)]
    else:
        results = []
        for item in folder_content:
            subfolder = os.path.join(folder, item)
            if os.path.isdir(subfolder):
                results += _traverse_run_folders(subfolder, traversed_directories + [item])
        return results


def results_from_folder_to_df(folder: str, metric_fn: typing.Callable) -> pd.DataFrame:
    """
    Traverses all subdirecties, and obtains stored runs with evaluation scores.

    Parameters
    ----------
    folder: str
        The folder that should be traversed (recursively)

    metric_fn: Callable
        Callable scikit-learn metric fn, see:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    Returns
    -------
    df : pd.DataFrame
        a data frame with columns for each folder level and the metric_fn as y
    """
    list_dirs_runs = _traverse_run_folders(folder, list())
    results = []
    for dirs, run in list_dirs_runs:
        current = {'folder_depth_%d' % idx: folder for idx, folder in enumerate(dirs)}
        scores = run.get_metric_fn(metric_fn)
        current['y'] = sum(scores) / len(scores)
        results.append(current)
    return pd.DataFrame(results)
