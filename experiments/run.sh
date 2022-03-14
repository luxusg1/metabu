#!/bin/bash

PYTHONPATH:/home/louisot/PycharmProjects/Metabu/experiments/externalPythonLib
export PYTHONPATH

python main.py  +task_id=3 \
                  +to_default=None \
                  searchParams.seed=1 \
                  searchParams.n_jobs=3 \
                  classifier=adaboost \
                  optimizerParams.name=random_sampling \
                  optimizerParams.num_iterations=20 \
                  searchParams.nb_neighbors_datasets=10 \
                  hydra.run.dir='data/outputs/'

