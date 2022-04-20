import collections
import csv
import openml
import openmlpimp
import os
import subprocess
import sys


def _determine_eligibility(strategy, include_pattern, exclude_pattern):
    if include_pattern is not None and not isinstance(include_pattern, list):
        raise TypeError()
    if exclude_pattern is not None and not isinstance(exclude_pattern, list):
        raise TypeError()

    # exlude strategies
    if exclude_pattern is not None:
        for pattern in exclude_pattern:
            if pattern in strategy:
                return False
    # include strategies
    if include_pattern is not None:
        for pattern in include_pattern:
            if pattern in strategy:
                return True

    # if include_pattern is None, all that was not excluded can be used.
    if include_pattern is None:
        return True
    else:
        return False


def _determine_name(strategy):
    strategy_splitted = strategy.split('__')
    if len(strategy_splitted) < 4:
        return strategy
    return strategy_splitted[0] + '__' + strategy_splitted[3]


def to_csv_file(ranks_dict, classifier, location):
    hyperparameters = None
    for task_id, params in ranks_dict.items():
        hyperparameters = set([openmlpimp.utils.name_mapping(classifier, param) for param in params.keys()])

    with open(location, 'w') as csvfile:
        fieldnames = ['task_id']
        fieldnames.extend(hyperparameters)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task_id, param_values in ranks_dict.items():
            result = {openmlpimp.utils.name_mapping(classifier, param): score for param, score in param_values.items()}
            result['task_id'] = 'Task %d' %task_id
            writer.writerow(result)
    pass


def to_csv_unpivot(ranks_dict, classifier, location):
    with open(location, 'w') as csvfile:
        fieldnames = ['task_id', 'param_id', 'param_name', 'variance_contribution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for task_id, param_values in ranks_dict.items():

            for param_name, variance_contribution in param_values.items():
                realname = openmlpimp.utils.name_mapping(classifier, param_name)
                result = {'task_id' : task_id,
                          'param_id': realname,
                          'param_name': realname,
                          'variance_contribution': variance_contribution}
                writer.writerow(result)
    pass


def plot_task(plotting_virtual_env, plotting_scripts_dir, strategy_directory, plot_directory, task_id, include_pattern=None, exclude_pattern=None):
    # make regret plot:
    script = "%s %s/plot_test_performance_from_csv.py " % (plotting_virtual_env, plotting_scripts_dir)
    cmd = [script]
    for strategy, directory in strategy_directory.items():
        if _determine_eligibility(strategy, include_pattern, exclude_pattern) is False:
            continue
        strategy_name = _determine_name(strategy)

        cmd.append(strategy_name)
        cmd.append(directory + '/' + str(task_id) + '/*' + '.csv')
    try:
        os.makedirs(plot_directory)
    except FileExistsError:
        pass

    cmd.append('--save %s ' % os.path.join(plot_directory, 'validation_regret%s.png' %str(task_id)))
    cmd.append('--ylabel "Accuracy Loss"')
    subprocess.run(' '.join(cmd), shell=True)
    print('CMD: ', ' '.join(cmd))


def boxplot_traces(strategy_traces, save_directory, name):
    # only import matplotlib for this function, as meta cluster doesn't like it
    import matplotlib.pyplot as plt
    try:
        os.makedirs(save_directory)
    except FileExistsError:
        pass

    data = []
    label_names = []

    for strategy, trace in strategy_traces.items():
        current = [trace_item.evaluation for trace_item in trace.trace_iterations.values()]
        data.append(current)
        label_names.append(strategy.split('__')[0])

    plt.figure()
    plt.boxplot(data)
    plt.xticks(list(range(1, len(label_names) + 1)), label_names)
    plt.savefig(os.path.join(save_directory, name))
    plt.close()


def average_rank(plotting_virtual_env, plotting_scripts_dir, output_directory, curves_directory, include_pattern=None, exclude_pattern=None, ylabel=None, xmax=50):
    script = "%s %s/plot_ranks_from_csv.py " % (plotting_virtual_env, plotting_scripts_dir)
    cmd = [script]

    results = collections.defaultdict(dict)
    for strategy in os.listdir(curves_directory):
        if _determine_eligibility(strategy, include_pattern, exclude_pattern) is False:
            continue
        strategy_name = _determine_name(strategy)

        for task_csv in os.listdir(os.path.join(curves_directory, strategy)):
            task_id = task_csv.split('.')[0]
            results[task_id][strategy_name] = os.path.join(curves_directory, strategy, task_csv)

    for task_id, strategy_file in results.items():
        for strategy, file in results[task_id].items():
            cmd.append('%s %s %s' % (str(task_id), strategy, file))

    filename = output_directory + '/average_ranks'
    if include_pattern:
        filename += '__inc__' + '__'.join(include_pattern)
    if exclude_pattern:
        filename += '__ex__' + '__'.join(exclude_pattern)
    cmd.append('--xlabel "Number of Iterations"')
    if ylabel:
        cmd.append('--ylabel "%s"' %ylabel)
    cmd.append('--save ' + filename + '.pdf')
    cmd.append('--xmin 0')
    cmd.append('--xmax %d' % xmax)
    print('CMD: ', ' '.join(cmd))
    subprocess.run(' '.join(cmd), shell=True)


def obtain_performance_curves(traces, save_directory, avg_curve_directory=None, identifier=None, improvements=True, inverse=False):
    def save_curve(current_curve, filename):
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['iteration', 'evaluation', 'evaluation2'])
            for idx in range(len(current_curve)):
                current_val = current_curve[idx]
                if inverse:
                    current_val = 1 - current_val
                if current_val < -0.0000001 or current_val > 1.0000001:
                    raise ValueError()
                csvwriter.writerow([idx+1, current_val, current_val])

    if isinstance(traces, openml.runs.OpenMLRunTrace):
        traces = [traces]

    curves = collections.defaultdict(dict)
    try:
        os.makedirs(save_directory)
    except FileExistsError:
        pass

    # each trace represents a different task (unique idx, although this is not the task_id)
    for idx, trace in enumerate(traces):
        for itt in trace.trace_iterations:
            cur = trace.trace_iterations[itt]
            curves[(idx, cur.repeat, cur.fold)][cur.iteration] = cur.evaluation

    for curve in curves.keys():
        current_curve = curves[curve]
        curves[curve] = list(collections.OrderedDict(sorted(current_curve.items())).values())

    if improvements:
        for curve in curves.keys():
            current_curve = curves[curve]
            for idx in range(1, len(current_curve)):
                if current_curve[idx] < current_curve[idx-1]:
                    current_curve[idx] = current_curve[idx - 1]

    if avg_curve_directory is not None:
        if identifier is None:
            raise ValueError()

        try:
            os.makedirs(avg_curve_directory)
        except FileExistsError:
            pass

        average_curve = [0] * len(curves[(0, 0, 0)])
        # assumes all curves have same nr of iterations :)
        for _, currentcurve in curves.items():
            for itt, value in enumerate(currentcurve):
                average_curve[itt] += value / len(curves)
        save_curve(average_curve, avg_curve_directory + '/%s.csv' % str(identifier))

    for idx, repeat, fold in curves.keys():
        save_curve(curves[(idx, repeat, fold)], save_directory + '/%d_%d_%d.csv' % (idx, repeat, fold))
