import collections
import openmlpimp
import os
import json


def obtain_marginal_contributions(result_directory):
    all_ranks = dict()
    all_tasks = list()
    total_ranks = None
    num_tasks = 0
    marginal_contribution = collections.defaultdict(list)

    for task_id in os.listdir(result_directory):
        task_dir = os.path.join(result_directory, task_id)
        if os.path.isdir(task_dir):
            pimp_file = os.path.join(task_dir, 'pimp_values_fanova.json')
            interaction_file = os.path.join(task_dir, 'pimp_values_fanova_interaction.json')

            if os.path.isfile(pimp_file) and os.path.isfile(interaction_file):
                hyperparameters = json.loads(open(pimp_file).read())
                hyperparameters.update(json.loads(open(interaction_file).read()))

                for hyperparameter, value in hyperparameters.items():
                    parts = hyperparameter.split('__')
                    if sorted(parts) != parts: continue

                    marginal_contribution[hyperparameter].append(value)
                all_tasks.append(task_id)

                all_ranks[task_id] = hyperparameters
                ranks = openmlpimp.utils.rank_dict(hyperparameters, reverse=True)
                if total_ranks is None:
                    total_ranks = ranks
                else:
                    total_ranks = openmlpimp.utils.sum_dict_values(total_ranks, ranks, allow_subsets=False)
                    num_tasks += 1
    total_ranks = openmlpimp.utils.divide_dict_values(total_ranks, num_tasks)
    return total_ranks, marginal_contribution, all_tasks