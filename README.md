# Metabu - Learning meta-features (Experiments)


## Setting up the environment

```bash
# install dependencies 
pip install -r requirements.txt
pip install -e .
# extract data
cd experiments
unzip data_metabu_iclr.zip
cd ..
```

Configuration details are defined under `conf/`.

## Task 1

```bash
you@machine$ python main.py --help

== Configuration groups ==
Compose your configuration from those groups (group=option)

metafeature: autosklearn, landmark, metabu, scot
pipeline: adaboost
task: task1, task2
```


