import os
import json
import tempfile
import openmlpimp

#from pimp.importance.importance import Importance


class PimpBackend(object):

    @staticmethod
    def execute(save_folder, runhistory_location, configspace_location, modus='ablation', seed=1):
        with open(runhistory_location, 'r') as runhistory_filep:
            runhistory = json.load(runhistory_filep)

        # create scenario file
        scenario_dict = {'run_obj': 'quality', 'deterministic': 1, 'paramfile': configspace_location}

        trajectory_lines = openmlpimp.utils.runhistory_to_trajectory(runhistory, maximize=True)
        if len(trajectory_lines) != 1:
            raise ValueError('trajectory file should containexactly one line.')

        traj_file = tempfile.NamedTemporaryFile('w', delete=False)
        for line in trajectory_lines:
            json.dump(line, traj_file)
            traj_file.write("\n")
        traj_file.close()

        num_params = len(trajectory_lines[0]['incumbent'])
        importance = Importance(scenario_dict,
                                runhistory_file=runhistory_location,
                                parameters_to_evaluate=num_params,
                                traj_file=traj_file.name,
                                seed=seed,
                                save_folder=save_folder)

        try: os.makedirs(save_folder)
        except FileExistsError: pass

        for i in range(5):
            try:
                result = importance.evaluate_scenario(modus)
                filename = 'pimp_values_%s.json' %modus
                with open(os.path.join(save_folder, filename), 'w') as out_file:
                    json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
                importance.plot_results(name=os.path.join(save_folder, modus), show=False)
                return save_folder + "/" + filename
            except ZeroDivisionError as e:
                pass
        raise e