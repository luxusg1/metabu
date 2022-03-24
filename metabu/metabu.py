import logging as log
import typing

import numpy as np
import pandas as pd
import torch

from metabu.fgw import train_fused_gromov_wasserstein
from metabu.utils import get_cost_matrix, intrinsic_estimator, get_pca_importances


class Metabu:
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
        return self.model.weight.detach().cpu().numpy()

    def predict(self, basic_reprs: pd.DataFrame) -> np.ndarray:
        return np.dot(basic_reprs[self.basic_repr_labels].values, self.psi.T)

    def train_predict(self,
                      basic_reprs: pd.DataFrame,
                      target_reprs: pd.DataFrame,
                      column_id: str,
                      test_ids: list,
                      train_ids: list) -> typing.Tuple[np.ndarray, np.ndarray]:
        basic_reprs_train = basic_reprs[basic_reprs[column_id].isin(train_ids)]
        basic_reprs_test = basic_reprs[basic_reprs[column_id].isin(test_ids)]
        target_reprs_train = target_reprs[target_reprs[column_id].isin(train_ids)]

        self.train(basic_reprs=basic_reprs, target_reprs=target_reprs_train, column_id=column_id)
        return self.predict(basic_reprs_train), self.predict(basic_reprs_test)

    def get_importances(self) -> typing.Tuple[np.ndarray, typing.List[str]]:
        imp = get_pca_importances(self.mds.embedding_)
        idx_best = imp.argmax()
        assert len(np.abs(self.psi[idx_best])) == len(self.basic_repr_labels)
        return np.abs(self.psi[idx_best]), self.basic_repr_labels
