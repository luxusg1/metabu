import logging as log
import typing

import numpy as np
import pandas as pd
import torch

from metabu.fgw import train_fused_gromov_wasserstein
from metabu.utils import get_cost_matrix, intrinsic_estimator, get_pca_importances


class Metabu:
    """Metabu: learning meta-features.

    Attributes
    ----------

    mds : sklearn.manifold.mds.MDS, default None
        Multi-dimensional scaling model trained afted step 2.
    intrinsic_dim : int, default None
        Intrinsic dimension of the benchmark, see last paragraph of section 4.
    psi : np.ndarray, default None
        Linear mapping psi of the basic representation to the metabu representation, learned during step 3.

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
        """

        Parameters
        ----------

        alpha: float, default 0.5,
            Trade-off parameter in FGW distance (eq. 1).
        lambda_reg: float, default 1e-3,
            L_1 regularization parameter (eq. 2).
        learning_rate: float, default 0.01,
            Learning rate used with ADAM optimizer.
        early_stopping_patience: int, default 10,
            Number of iterations without improvement.
        early_stopping_criterion_ndcg: int, default 10,
            Trunc value of NDCG, e.g. NDCG@10.
        verbose: bool, default True,
            Print verbose.
        ncpus: int, default 1,
            Number of cpu used, especially, to compute pairwise distance of the target representations.
        device: str, default "cpu",
            Device used by PyTorch ("cpu" or "gpu").
        seed: int, default 42
            Seed for reproducibility.

        """

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

        """Train the linear mapping psi.

        Parameters
        ----------

        basic_reprs : pandas.core.DataFrame
            Basic representations.

        target_reprs : pandas.core.DataFrame
            Target representations.

        column_id: str
            Name of the index column.

        Returns
        -------

        self

        """

        list_ids = sorted(list(basic_reprs[column_id].unique()))
        task_id_has_target_representation = target_reprs.task_id.unique()
        if set(list_ids) != set(task_id_has_target_representation):
            raise ValueError('Inconsistent numbers of instances.')

        basic_repr_labels = basic_reprs.columns
        self.basic_repr_labels = [_ for _ in basic_repr_labels if _ != column_id]
        log.info("Considering {0} basic meta-features: ".format(len(self.basic_repr_labels)) + ",".join(self.basic_repr_labels))

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
            device=self.device,
            list_ids=list_ids)
        return self

    @property
    def psi(self) -> np.ndarray:
        """Get the linear mapping psi.

        Returns
        -------

        psi : numpy.ndarray
            Weight matrix, representing the trained linear model.

        """
        return self.model.weight.detach().cpu().numpy()

    def predict(self, basic_reprs: pd.DataFrame) -> np.ndarray:
        """Predict the Metabu representations given basic representations.

        Parameters
        ----------

        basic_reprs : pandas.core.DataFrame
            Basic representations.

        Returns
        -------

        metabu_reprs : np.ndarray
            Metabu representations.

        """
        return np.dot(basic_reprs[self.basic_repr_labels].values, self.psi.T)

    def train_predict(self,
                      basic_reprs: pd.DataFrame,
                      target_reprs: pd.DataFrame,
                      column_id: str,
                      train_ids: list,
                      test_ids: list) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Learn the linear mapping psi using task instances in train_ids. Then predict Metabu representations separately for train and test instances.

        Parameters
        ----------

        basic_reprs : pandas.core.DataFrame
            Basic representations.

        target_reprs : pandas.core.DataFrame
            Target representations.

        column_id: str
            Name of the index column.

        train_ids : list of int
            List of training instances.

        test_ids : list of int
            List of testing instances.

        Returns
        -------

        metabu_train : numpy.ndarray
            Metabu representation of training instances.

        metabu_test : numpy.ndarray
            Metabu representation of testing instances.

        """

        basic_reprs_train = basic_reprs # [basic_reprs[column_id].isin(train_ids)]
        basic_reprs_test = basic_reprs[basic_reprs[column_id].isin(test_ids)]
        target_reprs_train = target_reprs # [target_reprs[column_id].isin(train_ids)]

        self.train(basic_reprs=basic_reprs_train, target_reprs=target_reprs_train, column_id=column_id)
        return self.predict(basic_reprs_train[basic_reprs[column_id].isin(train_ids)]), self.predict(basic_reprs_test)

    def get_importances(self) -> typing.Tuple[np.ndarray, typing.List[str]]:
        """Get the importance scores of each basic representation column.

        The scores are extracted from the trained linear mapping psi.

        Returns
        --------

        importances: list of float
            List of importance scores.

        list_labels: list of str
            List of corresponding basic representation column.

        """
        imp = get_pca_importances(self.mds.embedding_)
        idx_best = imp.argmax()
        assert len(np.abs(self.psi[idx_best])) == len(self.basic_repr_labels)
        return np.abs(self.psi[idx_best]), self.basic_repr_labels
