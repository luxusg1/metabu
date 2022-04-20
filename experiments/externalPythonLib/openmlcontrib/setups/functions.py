import collections
import ConfigSpace
import copy
import json
import openml
import typing

from openml.extensions.sklearn.extension import SklearnExtension


def filter_setup_list(setupid_setup, param_name, min=None, max=None, allowed_values=None):
    """
    Removes setups from a dict of setups if the parameter value
    does not comply with a given value. Important: Use with
    caution, as it does not handle duplicate names for various
    modules well. Will be updated in a non-backward compatible 
    manner in a later version. 
    
    Parameters
    ----------
    setupid_setup : dict of OpenMLSetup
        As obtained from the openml.setups.list_setups fn

    param_name : str
        the name of the parameter which values should be restricted

    min : int
        setups with values below this threshold will be removed

    max : int
        setups with values above this threshold will be removed

    allowed_values : list
        list of allowed values

    Returns
    -------
    model : dict of OpenMLSetup
        a dict, with the setups that did not comply removed
    """
    allowed = dict()

    for setupid, setup in setupid_setup.items():

        paramname_id = {v.parameter_name: k for k, v in setup.parameters.items()}

        if param_name not in paramname_id.keys():
            raise ValueError('Setup %d does not contain param %s' %(setupid, param_name))
        param_value = json.loads(setup.parameters[paramname_id[param_name]].value)

        if min is not None:
            if param_value < min:
                continue

        if max is not None:
            if param_value > max:
                continue

        if allowed_values is not None:
            if param_value not in allowed_values:
                continue

        allowed[setupid] = copy.deepcopy(setup)

    return allowed


def obtain_setups_by_ids(setup_ids, require_all=True, limit=250):
    """
    Obtains a list of setups by id. Because the

    Parameters
    ----------
    setup_ids : list[int]
        list of setups to obtain

    require_all : bool
        if set to True, the list of requested setups is required to
        be complete (and an error is thrown if not)

    limit : int
        The number of setups to obtain per call
        (lower numbers avoid long URLs, but take longer)

    Returns
    -------
    The dict of setups
    """
    if not isinstance(setup_ids, collections.Iterable):
        raise ValueError()
    for val in setup_ids:
        if not isinstance(val, int):
            raise ValueError()
    if len(setup_ids) == 0:
        raise ValueError('Please give at least one setup id.')

    setups = {}
    offset = 0
    setup_ids = list(setup_ids)
    while True:
        setups_batch = openml.setups.list_setups(setup=setup_ids[offset:offset+limit])
        setups.update(setups_batch)

        offset += limit
        if offset >= len(setup_ids):
            break
    if require_all:
        missing = set(setup_ids) - set(setups.keys())
        if set(setup_ids) != set(setups.keys()):
            raise ValueError('Did not get all setup ids. Missing: %s' % missing)

    return setups


def setup_to_parameter_dict(setup: openml.setups.OpenMLSetup,
                            flow: openml.flows.OpenMLFlow,
                            map_library_names: bool,
                            configuration_space: ConfigSpace.ConfigurationSpace):
    """
    Transforms a setup into a dict, containing the relevant parameters as key / value pair

    Parameters
    ----------
    setup : OpenMLSetup
        the OpenML setup object

    flow : OpenMLFlow
        the OpenML flow object. Should correspond to the setup

    map_library_names : bool
        whether to map the hyperparameter name to the libraries unique name (True) or OpenMLs unique name (False)

    configuration_space : ConfigurationSpace
        Configuration Space (used for determining relevant parameters and dependencies)

    Returns
    -------
    A dict mapping from parameter name to value

    """
    hyperparameter_values = dict()
    for pid, hyperparameter in setup.parameters.items():
        if map_library_names:
            name = SklearnExtension()._openml_param_name_to_sklearn(hyperparameter, flow)
        else:
            name = hyperparameter.fullName
        value = hyperparameter.value
        if name not in configuration_space.get_hyperparameter_names():
            continue

        if name in hyperparameter_values:
            # duplicate parameter name, this can happen due to sub-flows.
            # when this happens, we need to fix
            raise KeyError('Duplicate hyperparameter: %s' % name)
        hyperparameter_values[name] = json.loads(value)
    missing_parameters = set(configuration_space.get_hyperparameter_names()) - hyperparameter_values.keys()
    if len(missing_parameters) > 0:
        raise ValueError('Setup %d does not comply to relevant parameters set. Missing: %s' % (setup.setup_id,
                                                                                               str(missing_parameters)))
    config_to_check = ConfigSpace.Configuration(configuration_space, hyperparameter_values,
                                                allow_inactive_with_values=True)
    active_parameters = configuration_space.get_active_hyperparameters(config_to_check)

    for hyperparameter in set(hyperparameter_values.keys()):
        if hyperparameter not in active_parameters:
            del hyperparameter_values[hyperparameter]

    return hyperparameter_values


def setup_in_config_space(setup, flow, config_space):
    """
    Checks whether a given setup is within the boundaries of a config space

    Parameters
    ----------
    setup : OpenMLSetup
        the setup object

    flow : OpenMLFlow
        the OpenML flow object

    config_space : ConfigurationSpace
        The configuration space

    Returns
    -------
    Whether this setup is within the boundaries of a config space
    """
    # flow / setup ID checks are not super important, but nice as it doesn't add much overhead
    if flow.flow_id is None:
        raise ValueError('Flow does not contain an ID (please only use flows obtained from server)')
    if flow.flow_id != setup.flow_id:
        raise ValueError('Flow id does not correspond. Setup has %d, flow has %d' % (setup.flow_id, flow.flow_id))
    try:
        name_values = setup_to_parameter_dict(setup=setup,
                                              flow=flow,
                                              map_library_names=True,
                                              configuration_space=config_space)
        ConfigSpace.Configuration(config_space, name_values)
        return True
    except ValueError:
        return False


def filter_setup_list_by_config_space(setups: typing.Dict[int, openml.setups.OpenMLSetup],
                                      flow: openml.flows.OpenMLFlow,
                                      config_space: ConfigSpace.ConfigurationSpace):
    """
    Removes all setups that do not comply to the config space

    Parameters
    ----------
    setups : dict
        a dict mapping from setup id to the OpenMLSetup objects

    config_space : ConfigurationSpace
        The configuration space

    Returns
    -------
    A dict mapping from setup id to setup, all complying to the config space
    """
    if not isinstance(setups, dict):
        raise TypeError('setups should be of type: dict')

    setups_remain = {}
    for sid, setup in setups.items():
        if setup_in_config_space(setup, flow, config_space):
            setups_remain[sid] = setup
    return setups_remain
