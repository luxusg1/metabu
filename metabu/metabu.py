import logging as log
import typing

import numpy as np
import pandas as pd
import torch

from metabu.fgw import train_fused_gromov_wasserstein
from metabu.utils import get_cost_matrix, intrinsic_estimator, get_pca_importances


class Metabu:
    """
    Metabu

    Parameters
    ----------
    alpha: float, default 0.5,
        the trade-off parameter in fused_gromov_wasserstein distance
    lambda_reg: float, default 1e-3,
        the regularization weight
    learning_rate: float, default 0.01,
        parameter of ADAM optimizer
    early_stopping_patience: int, default 10,
        the training is stopped when successively early_stopping_patience no improvement are observed
    early_stopping_criterion_ndcg: int, default 10,
        Only consider the highest early_stopping_criterion_ndcg scores in the ranking when computing ndcg. If None, use all outputs.
    verbose: bool default True,
        output print during the training phase if set to True
    ncpus: int, default 1,
        number of cpu used to train the Linear Model
    device: str, choice:["cpu", "gpu"] default "cpu",
        used device
    seed: int, default 42
        variable for reproducibility




    Attributes
    ----------

    mds : sklearn.manifold._mds.MDS, default None
        multi dimensional scaling
    intrinsic_dim : int, default None
        the intrinsic dimension corresponding to the target representation
    model : torch.nn.Linear, default None
        The linear mapping of basic representation to metabu representation

    """

    def __init__(self,
                 alpha: float = 0.5,
                 lambda_reg: float = 1e-3,
                 learning_rate: float = 0.01,
                 early_stopping_patience: int = 10,
                 early_stopping_criterion_ndcg: int = 10,
                 verbose: bool = True,
                 ncpus: int = 1,
                 device: str = "cpu",
                 seed: int = 42) -> None:

        self.early_stopping_criterion_ndcg = early_stopping_criterion_ndcg
        self.seed = seed
        self.ncpus = ncpus
        self.verbose = verbose
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.mds = None
        self.intrinsic_dim = None
        self.device = torch.device(device)

        if verbose:
            log.basicConfig(format="%(asctime)s: %(message)s", level=log.DEBUG)
        else:
            log.basicConfig(format="%(asctime)s : %(message)s")

    def train(self,
              basic_reprs: pd.DataFrame,
              target_reprs: pd.DataFrame,
              column_id: str) -> None:

        """

        Train the Linear mapping of basic representation  to metabu representation

        :param basic_reprs: the basic representation
        :type basic_reprs: pandas.core.dataFrame

        :param target_reprs: the target representation
        :type target_reprs: pandas.core.dataFrame

        :param column_id: name of column which content the id of each datasets or tasks in the target_reprs dataframe
        :type column_id: str

        """

        list_ids = list(target_reprs[column_id].unique())

        task_id_has_target_representation = target_reprs.task_id.unique()
        basic_repr_labels = basic_reprs.columns
        self.basic_repr_labels = [_ for _ in basic_repr_labels if _ != column_id]

        if set(list_ids) != set(task_id_has_target_representation):
            raise ValueError('Inconsistent numbers of instances.')

        log.info("Compute pairwise distances of target representations.")
        cost_matrix = get_cost_matrix(target_repr=target_reprs, task_ids=list_ids, verbose=self.verbose)

        log.info("Compute intrinsic dimension.")
        self.intrinsic_dim = intrinsic_estimator(cost_matrix)

        log.info("Train Metabu meta-features.")
        self.model, self.mds = train_fused_gromov_wasserstein(
            basic_representations=basic_reprs.set_index(column_id),
            pairwise_dist_z=cost_matrix,
            learning_rate=self.learning_rate,
            seed=self.seed,
            early_stopping=self.early_stopping_patience,
            early_stopping_criterion_ndcg=self.early_stopping_criterion_ndcg,
            alpha=self.alpha,
            intrinsic_dim=self.intrinsic_dim,
            lambda_reg=self.lambda_reg,
            device=self.device, )

    @property
    def psi(self) -> np.ndarray:
        """
        Get the Linear mapping model weight as umpy array

        :return model_weight: the weight of the Linear model
        :rtype: np.ndarray

        """
        return self.model.weight.detach().cpu().numpy()

    def predict(self, basic_reprs: pd.DataFrame) -> np.ndarray:
        """
        predict the metabu representation corresponding to the given basic representation

        :param basic_reprs: The basic representation

        :return: metabu representation: the metabu representation

        """
        return np.dot(basic_reprs[self.basic_repr_labels].values, self.psi.T)

    def train_predict(self,
                      basic_reprs: pd.DataFrame,
                      target_reprs: pd.DataFrame,
                      column_id: str,
                      test_ids: list,
                      train_ids: list) -> typing.Tuple[np.ndarray, np.ndarray]:
        """

        Train the Linear mapping of basic representation to metabu representation using all task in train_ids and
        predict the metabu representation corresponding to the  basic representation for both tasks in test_ids and
        train_ids

        :param basic_reprs: the basic representation

        :param target_reprs: the target representation
        :param test_ids: list of test tasks (not use on the training step)
        :param test_ids: list of test tasks (not use on the training step)
        :param train_ids: list of train tasks (use on the training step)
        :param column_id: name of column which content the id of each datasets or tasks in the target_reprs dataframe

        :return metabu representation: metabu representation corresponding to the training and testing tasks
        """

        basic_reprs_train = basic_reprs[basic_reprs[column_id].isin(train_ids)]
        basic_reprs_test = basic_reprs[basic_reprs[column_id].isin(test_ids)]
        target_reprs_train = target_reprs[target_reprs[column_id].isin(train_ids)]

        self.train(basic_reprs=basic_reprs, target_reprs=target_reprs_train, column_id=column_id)
        return self.predict(basic_reprs_train), self.predict(basic_reprs_test)

    def get_importances(self) -> typing.Tuple[np.ndarray, typing.List[str]]:

        """

        Get the importance scores of each basic representation (each column of given the basic representation dataframe)
        according to the resulted metabu representation. More the scores of one basic representation is high more this
        basic representation is important for the concerned algorithm.

        :return importance: importance score and importance labels for each basic representation of task


        """
        imp = get_pca_importances(self.mds.embedding_)
        idx_best = imp.argmax()
        assert len(np.abs(self.psi[idx_best])) == len(self.basic_repr_labels)
        return np.abs(self.psi[idx_best]), self.basic_repr_labels
