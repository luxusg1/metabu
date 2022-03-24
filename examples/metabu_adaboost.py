import pandas as pd
from metabu import Metabu
from sklearn.preprocessing import StandardScaler

basic_representations = pd.read_csv("data/basic_representations.csv").fillna(0)
target_representations = pd.read_csv("./data/adaboost_target_representations.csv")
basic_representations = basic_representations[basic_representations.task_id.isin(target_representations.task_id.unique())]

# Normalize basic meta-features
basic_representations.loc[:, basic_representations.columns != 'task_id'] = StandardScaler().fit_transform(basic_representations.loc[:, basic_representations.columns != 'task_id'].values)

metabu = Metabu()
metabu.train(basic_reprs=basic_representations,
             target_reprs=target_representations,
             column_id="task_id")
metabu.predict(basic_reprs=basic_representations)
metabu.get_importances()