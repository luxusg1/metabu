import copy
import numpy as np
import openmlpimp
import random
import sklearn
import sys

#from openml.flows import flow_to_sklearn
from openml.extensions import get_extension_by_flow
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant


def obtain_classifier(configuration_space, indices, classifier=None, fixed_parameters=None):
        configuration = configuration_space.sample_configuration(1)
        if fixed_parameters is not None:
            while True:
                complies = True
                for parameter, value in fixed_parameters.items():
                    all = configuration.get_dictionary()
                    name = 'classifier:' + classifier + ':' + parameter
                    if all[name] != value:
                        complies = False
                if complies:
                    break
                # resample
                configuration = configuration_space.sample_configuration(1)
        classifier = openmlpimp.utils.config_to_classifier(configuration, indices)
        return classifier


def classifier_to_pipeline(classifier, indices):
    from openmlstudy14.preprocessing import ConditionalImputer
    steps = [('imputation', ConditionalImputer(strategy='median',
                                               fill_empty=0,
                                               strategy_nominal='most_frequent'),indices),
             ('hotencoding', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'),indices),
             ('scaling', sklearn.preprocessing.StandardScaler(with_mean=False)),
             ('variencethreshold', sklearn.feature_selection.VarianceThreshold()),
             ('classifier', classifier)]

    # TODO: Also scaling on Tree based models?
    if isinstance(classifier, RandomForestClassifier) or isinstance(classifier, AdaBoostClassifier):
        del steps[2]

    pipeline = sklearn.pipeline.Pipeline(steps=steps)
    return pipeline


def modeltype_to_classifier(model_type, params={}):
    required_params = dict()
    if model_type == 'adaboost':
        base_estimator_params = {}
        for param in list(params.keys()):
            if param.startswith('base_estimator__'):
                base_estimator_params[param[16:]] = params.pop(param)

        classifier = AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(**base_estimator_params), **params)
    elif model_type == 'decision_tree':
        classifier = sklearn.tree.DecisionTreeClassifier(**params)
    elif model_type == 'libsvm_svc':
        classifier = SVC(**params)
        required_params['classifier__probability'] = True
    elif model_type == 'sgd':
        classifier = sklearn.linear_model.SGDClassifier(**params)
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(**params)
    else:
        raise ValueError('Unknown classifier: %s' %model_type)
    return classifier, required_params


def config_to_classifier(config, indices):
    parameter_settings = config.get_dictionary()
    model_type = None
    pipeline_parameters = {}
    for param, value in parameter_settings.items():
        param_name = None
        splitted = param.split(':')
        if splitted[0] not in ['imputation', 'classifier']:
            raise ValueError()  # for now ..

        elif splitted[1] == '__choice__':
            if splitted[0] == 'classifier':
                model_type = value
            continue
        elif param == 'classifier:adaboost:max_depth':
            # exception ..
            param_name = 'classifier__base_estimator__max_depth'
        elif param == 'classifier:random_forest:max_features':
            # exception ..
            param_name = 'classifier__max_features'
            value = random.uniform(0.1, 0.9)
        else:
            # normal case
            param_name = splitted[0] + '__' + splitted[-1]

        if isinstance(value, str) and value == 'None':
            value = None

        if value == 'True':
            value = True
        elif value == 'False':
            value = False

        pipeline_parameters[param_name] = value
    if model_type is None:
        raise ValueError('Modeltype not recognized (set with classifier:__choice__ value)')

    classifier, required_parameters = modeltype_to_classifier(model_type)
    pipeline_parameters.update(required_parameters)

    pipeline = classifier_to_pipeline(classifier, indices)
    pipeline.set_params(**pipeline_parameters)
    return pipeline


def setups_to_configspace(setups,
                          default_params,
                          keyfield='parameter_name',
                          logscale_parameters=None,
                          ignore_parameters=None,
                          ignore_constants=True):
    # setups is result from openml.setups.list_setups call
    # note that this config space is not equal to the one
    # obtained from auto-sklearn; but useful for creating
    # the pcs file
    parameter_values = {}
    flow_id = None
    for setup_id in setups:
        current = setups[setup_id]
        if flow_id is None:
            flow_id = current.flow_id
        else:
            if current.flow_id != flow_id:
                raise ValueError('flow ids are expected to be equal. Expected %d, saw %s' %(flow_id, current.flow_id))

        for param_id in current.parameters.keys():
            name = getattr(current.parameters[param_id], keyfield)
            value = current.parameters[param_id].value
            if name not in parameter_values.keys():
                parameter_values[name] = set()
            parameter_values[name].add(value)

    uncovered = set(parameter_values.keys()) - set(default_params.keys())
    if len(uncovered) > 0:
        raise ValueError('Mismatch between keys default_params and parameter_values. Missing' %str(uncovered))

    def is_castable_to(value, type):
        try:
            type(value)
            return True
        except ValueError:
            return False

    cs = ConfigurationSpace()
    if logscale_parameters is None:
        logscale_parameters = set()
    # for parameter in logscale_parameters:
    #     if parameter not in parameter_values.keys():
    #         raise ValueError('(Logscale) Parameter not recognized: %s' %parameter)

    constants = set()
    for name in parameter_values.keys():
        if ignore_parameters is not None and name in ignore_parameters:
            continue

        all_values = parameter_values[name]
        if len(all_values) <= 1:
            constants.add(name)
            if ignore_constants:
                continue

        if all(is_castable_to(item, int) for item in all_values):
            all_values = [int(item) for item in all_values]
            lower = min(all_values)
            upper = max(all_values)
            default = default_params[name]
            if not is_castable_to(default, int):
                sys.stderr.write('Illegal default for parameter %s (expected int): %s' %(name, str(default)))
                default = int(lower + lower + upper / 2)

            hyper = UniformIntegerHyperparameter(name=name,
                                                 lower=lower,
                                                 upper=upper,
                                                 default=default,
                                                 log=name in logscale_parameters)
            cs.add_hyperparameter(hyper)
        elif all(is_castable_to(item, float) for item in all_values):
            all_values = [float(item) for item in all_values]
            lower = min(all_values)
            upper = max(all_values)
            default = default_params[name]
            if not is_castable_to(default, float):
                sys.stderr.write('Illegal default for parameter %s (expected int): %s' %(name, str(default)))
                default = lower + lower + upper / 2

            hyper = UniformFloatHyperparameter(name=name,
                                               lower=lower,
                                               upper=upper,
                                               default=default,
                                               log=name in logscale_parameters)
            cs.add_hyperparameter(hyper)
        else:
            values = [get_extension_by_flow(item).flow_to_model(item) for item in all_values]
            hyper = CategoricalHyperparameter(name=name,
                                              choices=values,
                                              default=default_params[name])
            cs.add_hyperparameter(hyper)
    return cs, constants


def reverse_runhistory(runhistory):
    for idx in range(len(runhistory['data'])):
        score = runhistory['data'][idx][1][0]
        if score < 0.0:
            raise ValueError('score should be >= 0.0')
        if score > 1.0:
            raise ValueError('score should be <= 1.0')
        runhistory['data'][idx][1][0] = 1.0 - score


def scale_configspace_to_log(configspace):
    configspace_prime = ConfigurationSpace()
    for hyperparameter in configspace.get_hyperparameters():
        if isinstance(hyperparameter, CategoricalHyperparameter):
            prime = copy.deepcopy(hyperparameter)
            configspace_prime.add_hyperparameter(prime)
        elif isinstance(hyperparameter, UniformIntegerHyperparameter) or isinstance(hyperparameter, UniformFloatHyperparameter):
            if hyperparameter.log:
                lower = np.log(hyperparameter.lower)
                upper = np.log(hyperparameter.upper)
                default = np.log(hyperparameter.default_value)
                prime = UniformFloatHyperparameter(name=hyperparameter.name, lower=lower, upper=upper, default_value=default, log=False)
                configspace_prime.add_hyperparameter(prime)
            else:
                prime = copy.deepcopy(hyperparameter)
                configspace_prime.add_hyperparameter(prime)
        else:
            raise ValueError()
    return configspace_prime


def runhistory_to_trajectory(runhistory, maximize):
    trajectory_lines = []
    lowest_cost = None
    lowest_cost_idx = None
    highest_cost = None
    highest_cost_index = None

    all_costs = set()
    for run in runhistory['data']:
        config_id = run[0][0] # magic index
        cost = run[1][0] # magic index
        all_costs.add(cost)
        if lowest_cost is None or cost < lowest_cost:
            lowest_cost = cost
            lowest_cost_idx = config_id
        if highest_cost is None or cost > highest_cost:
            highest_cost = cost
            highest_cost_index = config_id

    if lowest_cost == highest_cost:
        raise ValueError('Lowest cost == highst cost. No ablation possible. ')

    def _default_trajectory_line():
        return {"cpu_time": 0.0, "evaluations": 0, "total_cpu_time": 0.0, "wallclock_time": 0.0}

    def paramdict_to_incumbent(param_dict):
        res = []
        for param in param_dict.keys():
            res.append(param + "='" + str(param_dict[param]) + "'")
        return res

    final = _default_trajectory_line()
    if maximize:
        final['cost'] = highest_cost
        final['incumbent'] = paramdict_to_incumbent(runhistory['configs'][str(highest_cost_index)])
    else:
        final['cost'] = lowest_cost
        final['incumbent'] = paramdict_to_incumbent(runhistory['configs'][str(lowest_cost_idx)])
    trajectory_lines.append(final)

    return trajectory_lines
