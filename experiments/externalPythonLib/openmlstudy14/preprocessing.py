#from sklearn.preprocessing.imputation import Imputer, _get_mask
from sklearn.impute import SimpleImputer as Imputer

from sklearn.feature_selection import VarianceThreshold

import warnings
import math
import numpy as np
from scipy import sparse

from sklearn.utils import check_array
#from sklearn.utils.fixes import astype
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES


class ConditionalImputer(Imputer):
    """Imputation transformer for completing missing values.
    Read more in the :ref:`User Guide <imputation>`.
    Parameters
    ----------
    missing_values : integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".
    strategy : string, optional (default="mean")
        The imputation strategy.
        - If "mean", then replace missing values using the mean along
          the axis.
        - If "median", then replace missing values using the median along
          the axis.
        - If "most_frequent", then replace missing using the most frequent
          value along the axis.
    strategy_nominal : string, optional (default="most_frequent")
        The imputation strategy for nominal attributes. For values, see "strategy"
    indices_nominal : list (int)
        An array of indices determining which are treated as nominal. If None,
        the Conditional Imputer will guess based on the values
    axis : integer, optional (default=0)
        The axis along which to impute.
        - If `axis=0`, then impute along columns.
        - If `axis=1`, then impute along rows. (Not supported)
    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.
    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:
        - If X is not an array of floating values;
        - If X is sparse and `missing_values=0`;
        - If `axis=0` and X is encoded as a CSR matrix;
        - If `axis=1` and X is encoded as a CSC matrix.
    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature if axis == 0.
    Notes
    -----
    - When ``axis=0``, columns which only contained missing values at `fit`
      are discarded upon `transform`.
    - When ``axis=1``, an exception is raised if there are rows for which it is
      not possible to fill in the missing values (e.g., because they only
      contain missing values).
    """
    #*******"NAN" changed to np.nan on construction default value of mising_values
    def __init__(self, missing_values=np.nan, strategy="mean",
                 strategy_nominal="most_frequent",
                 categorical_features=None,
                 fill_empty=None,
                 axis=0, verbose=0, copy=True, add_indicator=False):

        self.missing_values = missing_values
        self.strategy = strategy
        self.strategy_nominal = strategy_nominal
        self.categorical_features = categorical_features
        self.categorical_features_implied = None
        self.fill_empty = fill_empty
        self.axis = axis
        self.verbose = verbose
        self.copy = copy

        #***************************** changed by luxuxs to debug
        self.add_indicator = add_indicator
        #*****************************

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.axis not in [0]:
            raise ValueError(
                "Can only impute missing values on axis 0 (axis 1 not supported), "
                " got axis={0}".format(self.axis))

        X = check_array(X, accept_sparse='csc', dtype=np.float64,
                        force_all_finite=False)

        if sparse.issparse(X):
            statistics_general = self._sparse_fit(X,
                                                  self.strategy,
                                                  self.missing_values,
                                                  self.axis)
            statistics_nominal = self._sparse_fit(X,
                                                  self.strategy_nominal,
                                                  self.missing_values,
                                                  self.axis)
        else:
            statistics_general = self._dense_fit(X,
                                                 self.strategy,
                                                 self.missing_values,
                                                 self.axis)
            statistics_nominal = self._dense_fit(X,
                                                 self.strategy_nominal,
                                                 self.missing_values,
                                                 self.axis)

        # here the indexes of nominal values get set
        self.statistics_ = statistics_general
        if self.categorical_features is not None:
            for i in self.categorical_features:
                self.statistics_[i] = statistics_nominal[i]
        else:
            # iterate over all attributes
            self.categorical_features_implied = []
            for iAtt in range(len(statistics_general)):
                isNominal = True
                for iInst in range(len(X)):
                    if not np.isnan(X[iInst][iAtt]) and math.floor(
                            X[iInst][iAtt]) != X[iInst][iAtt]:
                        isNominal = False
                        break
                if isNominal:
                    # book keeping, for testing purposes
                    self.categorical_features_implied.append(iAtt)
                    self.statistics_[iAtt] = statistics_nominal[iAtt]

        if self.fill_empty is not None:
            self.statistics_[np.isnan(self.statistics_)] = self.fill_empty

        return self

    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
        """
        if self.axis == 0:
            check_is_fitted(self, 'statistics_')
            X = check_array(X, accept_sparse='csc', dtype=FLOAT_DTYPES,
                            force_all_finite=False, copy=self.copy)
            statistics = self.statistics_.copy()
            if X.shape[1] != statistics.shape[0]:
                raise ValueError("X has %d features per sample, expected %d"
                                 % (X.shape[1], self.statistics_.shape[0]))

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data need to be recomputed
        # when the imputation is done per sample
        else:
            X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                            force_all_finite=False, copy=self.copy)

            if sparse.issparse(X):
                statistics = self._sparse_fit(X,
                                              self.strategy,
                                              self.missing_values,
                                              self.axis)

            else:
                statistics = self._dense_fit(X,
                                             self.strategy,
                                             self.missing_values,
                                             self.axis)

        # Delete the invalid rows/columns
        invalid_mask = np.isnan(statistics)
        valid_mask = np.logical_not(invalid_mask)
        valid_statistics = statistics[valid_mask]
        valid_statistics_indexes = np.where(valid_mask)[0]
        missing = np.arange(X.shape[not self.axis])[invalid_mask]

        if self.axis == 0 and invalid_mask.any():
            if self.verbose:
                warnings.warn("Deleting features without "
                              "observed values: %s" % missing)
            X = X[:, valid_statistics_indexes]
        elif self.axis == 1 and invalid_mask.any():
            raise ValueError("Some rows only contain "
                             "missing values: %s" % missing)

        # Do actual imputation
        if sparse.issparse(X) and self.missing_values != 0:
            mask = _get_mask(X.data, self.missing_values)
            indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
                                np.diff(X.indptr))[mask]

            #X.data[mask] = astype(valid_statistics[indexes], X.dtype,
            #                      copy=False)

            X.data[mask] = valid_statistics.astype(X.dtype,copy=False)
        else:
            if sparse.issparse(X):
                X = X.toarray()

            if self.missing_values == "NaN" or np.isnan(self.missing_values):
                mask = np.isnan(X)
            else:
                mask =  X == self.missing_values

            #mask = _get_mask(X, self.missing_values)
            
            
            n_missing = np.sum(mask, axis=self.axis)
            values = np.repeat(valid_statistics, n_missing)

            if self.axis == 0:
                coordinates = np.where(mask.transpose())[::-1]
            else:
                coordinates = mask

            X[coordinates] = values

        return X


class MemoryEfficientVarianceThreshold(VarianceThreshold):
    """Features selector that removes all low-variance features.

    Subclass of ``sklearn.feature_selection.VarianceThreshold``. Differs only
    that calls ``np.var`` column-wise instead of on the whole array and does
    not accept any sparse matrix
    ."""

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        X = check_array(X)

        self.variances_ = []
        for i in range(X.shape[1]):
            self.variances_.append(np.var(check_array(X[:, i].reshape((-1, 1)),
                                                      dtype=np.float64)))
        self.variances_ = np.array(self.variances_)

        if np.all(self.variances_ <= self.threshold):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        return self