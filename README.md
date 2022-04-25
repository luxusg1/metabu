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

```bash
you@machine$ python main.py --help

== Configuration groups ==
Compose your configuration from those groups (group=option)

metafeature: autosklearn, landmark, metabu, scot
pipeline: adaboost, random_forest, svm
task: task1, task2
```

## Task 1

```bash
$ python main.py task=task1 openml_tid=3 task.ndcg=15 metafeature=metabu data_path=${PWD}/experiments/data_metabu_iclr pipeline=adaboost     
Task 1: 
- pipeline: adaboost 
- Metafeature: metabu 
- OpenML task: 3 
- NDCG@15: 0.6666666666666666
$ python main.py task=task1 openml_tid=3 task.ndcg=15 metafeature=autosklearn data_path=${PWD}/experiments/data_metabu_iclr pipeline=adaboost
Task 1: 
- pipeline: adaboost 
- Metafeature: AutoSklearn 
- OpenML task: 3 
- NDCG@15: 0.4
```

## Task 2

```bash
(metabu_exp) [hrakotoa@marg007 metabu]$ python main.py task=task2 task.nb_iterations=5 openml_tid=3 metafeature=metabu pipeline=random_forest data_path=${PWD}/experiments/data_metabu_iclr/
Iter=1
         hp={'classifier__bootstrap': True, 'classifier__criterion': 'entropy', 'classifier__max_features': 0.1690449621849076, 'classifier__min_samples_leaf': 11, 'classifier__min_samples_split': 3, 'imputation__strategy': 'mean'}
         perf=0.972148315047022
Iter=2
         hp={'classifier__bootstrap': True, 'classifier__criterion': 'entropy', 'classifier__max_features': 0.363192903976134, 'classifier__min_samples_leaf': 3, 'classifier__min_samples_split': 9, 'imputation__strategy': 'most_frequent'}
         perf=0.9921757445141066
Iter=3
         hp={'classifier__bootstrap': True, 'classifier__criterion': 'gini', 'classifier__max_features': 0.5, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'imputation__strategy': 'mean'}
         perf=0.9953027037617554
Iter=4
         hp={'classifier__bootstrap': True, 'classifier__criterion': 'gini', 'classifier__max_features': 0.3359939836076053, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 13, 'imputation__strategy': 'mean'}
         perf=0.9943652037617555
Iter=5
         hp={'classifier__bootstrap': True, 'classifier__criterion': 'entropy', 'classifier__max_features': 0.2787811513455088, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 3, 'imputation__strategy': 'mean'}
         perf=0.9953046630094045
```


