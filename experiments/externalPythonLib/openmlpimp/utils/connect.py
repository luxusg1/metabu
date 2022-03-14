import copy
import openml
import openmlpimp
import openmlcontrib

import os
import json

from openml.exceptions import OpenMLServerException

from ConfigSpace.read_and_write.pcs_new import write


def task_counts(flow_id):
    task_ids = {}
    offset = 0
    limit = 10000
    while True:
        try:
            runs = openml.runs.list_runs(flow=[flow_id], size=limit, offset=offset)
        except OpenMLServerException:
            runs = {}

        for run_id, run in runs.items():
            task_id = run['task_id']
            if task_id not in task_ids:
                task_ids[task_id] = 0
            task_ids[task_id] += 1
        if len(runs) < limit:
            break
        else:
            offset += limit
    return task_ids


def obtain_runhistory_and_configspace(flow_id, task_id,
                                      model_type,
                                      keyfield='parameter_name',
                                      required_setups=None,
                                      fixed_parameters=None,
                                      ignore_parameters=None,
                                      reverse=False):

    
    #from smac.tae.execute_ta_run import StatusType
    from smac.tae.execute_ta_run_old import StatusType
    all_fixed_parameters = copy.deepcopy(ignore_parameters)
    if fixed_parameters is not None:
        all_fixed_parameters.update(fixed_parameters)

    

    #config_space = openmlpimp.utils.get_config_space_casualnames(model_type, all_fixed_parameters)changed by luxus
    config_space = openmlpimp.configspaces.get_config_space(model_type,10)
    #openmlpimp.configspaces.get_config_space('adaboost',None)
    print("****mbola ********")
    valid_hyperparameters = config_space._hyperparameters.keys()

    evaluations = openml.evaluations.list_evaluations(function="predictive_accuracy", flows=[flow_id], tasks=[task_id])
    setup_ids = set()

    
    for run_id in evaluations.keys():
        setup_ids.add(evaluations[run_id].setup_id)

    if required_setups is not None:
        if len(setup_ids) < required_setups:
            raise ValueError('Not enough (evaluated) setups found on OpenML. Found %d; required: %d' %(len(setup_ids), required_setups))

    setups = openmlcontrib.setups.obtain_setups_by_ids(setup_ids)
    
    if fixed_parameters is not None:
        for param, value in fixed_parameters.items():
            print('restricting', param, value)
            setups = openmlcontrib.setups.filter_setup_list(setups, param, allowed_values=[value])
    
    print('Setup count; before %d after %d' %(len(setup_ids), len(setups)))
    setup_ids = set(setups.keys())

    # filter again ..
    if required_setups is not None:
        if len(setup_ids) < required_setups:
            raise ValueError('Not enough (evaluated) setups left after filtering. Got %d; required: %d' %(len(setup_ids), required_setups))

    data = []
    configs = {}
    applicable_setups = set()
    for run_id in evaluations.keys():
        config_id = evaluations[run_id].setup_id
        if config_id in setup_ids:
            if not openmlcontrib.setups.setup_in_config_space(setups[config_id], config_space=config_space,flow = openml.flows.get_flow(flow_id)):
                continue

            cost = evaluations[run_id].value
            runtime = 0.0 # not easily accessible
            status = {"__enum__": str(StatusType.SUCCESS)}
            additional = {}
            performance = [cost, runtime, status, additional]

            instance = openml.config.server + "task/" + str(task_id)
            seed = 1  # not relevant
            run = [config_id, instance, seed]

            applicable_setups.add(config_id)
            data.append([run, performance])

    for setup_id in applicable_setups:
        config = {}
        for param_id in setups[setup_id].parameters:
            name = getattr(setups[setup_id].parameters[param_id], keyfield)
            value = openml.extensions.get_extension_by_flow(openml.flows.get_flow(flow_id)).flow_to_model(setups[setup_id].parameters[param_id].value)
            if ignore_parameters is not None and name in ignore_parameters:
                continue
            if fixed_parameters is not None and name in fixed_parameters:
                continue
            if name not in valid_hyperparameters:
                continue
            # TODO: hack
            if isinstance(value, bool):
                value = str(value)
            config[name] = value
        configs[setup_id] = config

    run_history = {"data": data, "configs": configs}

    if reverse:
        openmlpimp.utils.reverse_runhistory(run_history)

    return run_history, config_space


def cache_runhistory_configspace(save_folder, flow_id, task_id, model_type, required_setups, reverse=False, fixed_parameters=None, ignore_parameters=None):
    if fixed_parameters:
        save_folder_suffix = [param + '_' + value for param, value in fixed_parameters.items()]
        save_folder_suffix = '/' + '__'.join(save_folder_suffix)
    else:
        save_folder_suffix = '/vanilla'

    runhistory_path = save_folder + save_folder_suffix + '/runhistory.json'
    configspace_path = save_folder + save_folder_suffix + '/config_space.pcs'
    print(runhistory_path, configspace_path)

    if not os.path.isfile(runhistory_path) or not os.path.isfile(configspace_path):
        runhistory, configspace = openmlpimp.utils.obtain_runhistory_and_configspace(flow_id, task_id, model_type,
                                                                                     required_setups=required_setups,
                                                                                     fixed_parameters=fixed_parameters,
                                                                                     ignore_parameters=ignore_parameters,
                                                                                     reverse=reverse)

        os.makedirs(save_folder + save_folder_suffix, exist_ok=True)

        with open(runhistory_path, 'w') as outfile:
            json.dump(runhistory, outfile, indent=2)

        with open(configspace_path, 'w') as outfile:
            outfile.write(write(configspace))
    else:
        print('[Obtained from cache]')

    # now the files are guaranteed to exists
    return runhistory_path, configspace_path
