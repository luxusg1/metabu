import os
import json
import openmlpimp
import ConfigSpace

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ConfigSpace.read_and_write.pcs_new import read
from fanova.fanova import fANOVA as fanova_pyrfr
from fanova.visualizer import Visualizer


class FanovaBackend(object):

    @staticmethod
    def _plot_result(fANOVA, configspace, directory, yrange=None):
        os.makedirs(directory, exist_ok=True)
        vis = Visualizer(fANOVA, configspace, directory, y_label='Predictive Accuracy')

        for hp1 in configspace.get_hyperparameters():
            plt.close('all')
            plt.clf()
            param = hp1.name
            outfile_name = os.path.join(directory, param.replace(os.sep, "_") + ".pdf")
            vis.plot_marginal(configspace.get_idx_by_hyperparameter_name(param), show=False)

            x1, x2, _, _ = plt.axis()
            if yrange:
                plt.axis((x1, x2, yrange[0], yrange[1]))
            plt.savefig(outfile_name)

        pass

    @staticmethod
    def execute(save_folder, runhistory_location, configspace_location, manual_logtransform, use_percentiles, interaction_effect, n_trees, run_limit=None, draw_plots=True):

        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = True

        with open(runhistory_location) as runhistory_file:
            runhistory = json.load(runhistory_file)
        with open(configspace_location) as configspace_file:
            configspace = read(configspace_file)
        os.makedirs(save_folder, exist_ok=True)

        X = []
        y = []

        for item in runhistory['data']:
            if run_limit is not None and len(X) > run_limit:
                break

            valid = True
            current = []
            setup_id = str(item[0][0])
            configuration = runhistory['configs'][setup_id]
            for param in configspace.get_hyperparameters():
                print(configuration)
                value = configuration[param.name]
                
                if isinstance(param, ConfigSpace.hyperparameters.UniformFloatHyperparameter) and not isinstance(value, float):
                    valid = False
                elif isinstance(param, ConfigSpace.hyperparameters.UniformIntegerHyperparameter) and not isinstance(value, int):
                    valid = False

                if isinstance(param, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                    value = param.choices.index(value)
                elif param.log and manual_logtransform:
                    value = np.log(value)

                current.append(value)
            if valid:
                X.append(current)
                y.append(item[1][0])
            else:
                print('Illegal configuration', current)
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError('Wrong shape')

        if manual_logtransform:
            configspace = openmlpimp.utils.scale_configspace_to_log(configspace)

        cutoffs = (-np.inf, np.inf)
        if use_percentiles:
            p75 = np.percentile(y, 75.0)
            p100 = np.percentile(y, 100.0)
            cutoffs = (p75, p100)

        # start the evaluator
        evaluator = fanova_pyrfr(X=X, Y=y, config_space=configspace, config_on_hypercube=False, cutoffs=cutoffs, n_trees=n_trees)
        # obtain the results
        params = configspace.get_hyperparameters()
        result = {}

        for idx, param in enumerate(params):
            importance = evaluator.quantify_importance([idx])[(idx,)]['total importance']
            result[param.name] = importance

        # store main results to disk
        filename = 'pimp_values_fanova.json'
        with open(os.path.join(save_folder, filename), 'w') as out_file:
            json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
            print('Saved individuals to %s' %os.path.join(save_folder, filename))


        # call plotting fn
        yrange = (0, 1)
        if use_percentiles:
            yrange = (p75, p100)
        if draw_plots:
            FanovaBackend._plot_result(evaluator, configspace, save_folder + '/fanova', yrange)

        if interaction_effect:
            result_interaction = {}
            for idx, param in enumerate(params):
                for idx2, param2 in enumerate(params):
                    if param.name >= param2.name: # string comparison cause stable
                        continue
                    print('interaction effects between', param.name, param2.name)
                    interaction = evaluator.quantify_importance([idx, idx2])[(idx,idx2)]['total importance']
                    interaction -= result[param.name]
                    interaction -= result[param2.name]
                    combined_name = param.name + '__' + param2.name
                    if interaction < 0.0:
                        raise ValueError('interaction score too low. Params: %s score %d' %(combined_name, interaction))
                    result_interaction[combined_name] = interaction

            for idx, param in enumerate(params):
                for idx2, param2 in enumerate(params):
                    if param.name >= param2.name:  # string comparison cause stable
                        continue
                    for idx3, param3 in enumerate(params):
                        if param2.name >= param3.name:  # string comparison cause stable
                            continue

                        print('interaction effects between', param.name, param2.name, param3.name)
                        interaction = evaluator.quantify_importance([idx, idx2, idx3])[(idx, idx2, idx3)]['total importance']
                        interaction -= result[param.name]
                        interaction -= result[param2.name]
                        interaction -= result[param3.name]
                        combined_name = param.name + '__' + param2.name + '__' + param3.name

                        interaction -= result_interaction[param.name + '__' + param2.name]
                        interaction -= result_interaction[param2.name + '__' + param3.name]
                        interaction -= result_interaction[param.name + '__' + param3.name]

                        if interaction < 0.0:
                            raise ValueError('interaction score too low. Params: %s score %d' % (combined_name, interaction))
                        result_interaction[combined_name] = interaction

            # store interaction effects to disk

            if sum(result_interaction.values()) + sum(result.values()) > 1:
                raise ValueError('Sum of results too high')

            filename = 'pimp_values_fanova_interaction.json'
            with open(os.path.join(save_folder, filename), 'w') as out_file:
                json.dump(result_interaction, out_file, sort_keys=True, indent=4, separators=(',', ': '))
                print('Saved interactions to %s' %os.path.join(save_folder, filename))
            if draw_plots:
                vis = Visualizer(evaluator, configspace, save_folder + '/fanova', y_label='Predictive Accuracy')
                vis.create_most_important_pairwise_marginal_plots()

        return save_folder + "/" + filename
