import random
import numpy as np
import scipy

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from openmlstudy14.distributions import loguniform, loguniform_int
from openmlstudy14.preprocessing import ConditionalImputer, MemoryEfficientVarianceThreshold


class EstimatorFactory():

    def __init__(self, n_folds_inner_cv=3, n_iter=200, n_jobs=-1):
        scoring = 'accuracy'
        error_score = 'raise'
        self.grid_arguments = dict(scoring=scoring,
                                   error_score=error_score,
                                   cv=n_folds_inner_cv,
                                   n_jobs=n_jobs)
        self.rs_arguments = dict(scoring=scoring,
                                 error_score=error_score,
                                 cv=n_folds_inner_cv,
                                 n_iter=n_iter,
                                 n_jobs=n_jobs,
                                 verbose=10)

        self.all_estimators = [self.get_naive_bayes,
                               self.get_decision_tree,
                               self.get_logistic_regression,
                               self.get_gradient_boosting,
                               self.get_svm,
                               self.get_random_forest,
                               self.get_mlp,
                               self.get_knn]

        self.estimator_mapping = {'naive_bayes': self.get_naive_bayes,
                                  'decision_tree': self.get_decision_tree,
                                  'logreg':
                                      self.get_logistic_regression,
                                  'gradient_boosting':
                                      self.get_gradient_boosting,
                                  'svm': self.get_svm,
                                  'random_forest': self.get_random_forest,
                                  'mlp': self.get_mlp,
                                  'knn': self.get_knn}

    @staticmethod
    def _get_pipeline(nominal_indices, sklearn_model):
        if len(nominal_indices) > 0:
            with_mean = False
        else:
            with_mean = True

        steps = [('Imputer', ConditionalImputer(strategy='median',
                                                fill_empty=0,
                                                categorical_features=nominal_indices,
                                                strategy_nominal='most_frequent')),
                 ('OneHotEncoder', OneHotEncoder(sparse=True,
                                                 handle_unknown='ignore',
                                                 categorical_features=nominal_indices)),
                 ('VarianceThreshold', VarianceThreshold()),
                 ('Scaler', StandardScaler(with_mean=with_mean)),
                 ('Estimator', sklearn_model)]
        return Pipeline(steps=steps)


    def get_naive_bayes(self, nominal_indices, **kwargs):
        return self._get_pipeline(nominal_indices, GaussianNB())


    def get_decision_tree(self, nominal_indices, **kwargs):
        param_dist = {'Estimator__min_samples_split': loguniform_int(base=2, low=2**1, high=2**7),
                      'Estimator__min_samples_leaf': loguniform_int(base=2, low=2**0, high=2**6)}
        decision_tree = self._get_pipeline(nominal_indices, DecisionTreeClassifier())
        random_search = RandomizedSearchCV(decision_tree,
                                           param_dist,
                                           **self.rs_arguments)
        return random_search


    def get_svm(self, nominal_indices, **kwargs):
        param_dist = {'Estimator__C': loguniform(base=2, low=2**-12, high=2**12),
                      'Estimator__gamma': loguniform(base=2, low=2**-12, high=2**12)}
        svm = self._get_pipeline(nominal_indices, SVC(kernel='rbf',probability=True))
        random_search = RandomizedSearchCV(svm,
                                           param_dist,
                                           **self.rs_arguments)
        return random_search


    def get_gradient_boosting(self, nominal_indices, **kwargs):
        param_dist = {'Estimator__learning_rate': loguniform(base=10, low=10**-4, high=10**-1),
                      'Estimator__max_depth': scipy.stats.randint(1, 5 + 1),
                      'Estimator__n_estimators': scipy.stats.randint(500, 10001)}
        boosting = self._get_pipeline(nominal_indices, GradientBoostingClassifier())
        random_search = RandomizedSearchCV(boosting,
                                           param_dist,
                                           **self.rs_arguments)
        return random_search


    def get_knn(self, nominal_indices, **kwargs):
        param_dist = {'Estimator__n_neighbors': list(range(1, 50 + 1))}
        knn = self._get_pipeline(nominal_indices, KNeighborsClassifier())
        grid_search = GridSearchCV(knn, param_dist, **self.grid_arguments)
        return grid_search


    def get_mlp(self, nominal_indices, **kwargs):
        param_dist = {'Estimator__hidden_layer_sizes': loguniform_int(base=2, low=2**5, high=2**11),
                      'Estimator__learning_rate_init': loguniform(base=10, low=10**-5, high=10**0),
                      'Estimator__alpha': loguniform(base=10, low=10**-7, high=10**-4),
                      'Estimator__max_iter': scipy.stats.randint(2, 1001),
                      'Estimator__momentum': scipy.stats.uniform(loc=0.1, scale=0.8)}
        mlp = self._get_pipeline(nominal_indices, MLPClassifier(activation='tanh', solver='adam'))
        random_search = RandomizedSearchCV(mlp, param_dist, **self.rs_arguments)
        return random_search


    def get_logistic_regression(self, nominal_indices, **kwargs):
        param_dist = {'Estimator__C': loguniform(base=2, low=2**-12, high=2**12)}
        logreg = self._get_pipeline(nominal_indices, LogisticRegression())
        random_search = RandomizedSearchCV(logreg, param_dist, **self.rs_arguments)
        return random_search


    def get_random_forest(self, nominal_indices, **kwargs):
        num_features = kwargs['num_features']
        lower = np.power(num_features, 0.1)
        upper = np.power(num_features, 0.9)
        scale = upper - lower
        lower = lower / num_features
        scale = scale / num_features
        param_dist = {'Estimator__max_features': scipy.stats.uniform(loc=lower, scale=scale)}
        randomforest = self._get_pipeline(nominal_indices, RandomForestClassifier(n_estimators=500))
        grid_search = RandomizedSearchCV(randomforest, param_dist, **self.rs_arguments)
        return grid_search


    def get_random_estimator(self, nominal_indices, **kwargs):
        estimator = random.choice(self.all_estimators)
        return estimator(nominal_indices, **kwargs)


    def get_all_flows(self, nominal_indices, **kwargs):
        return [estimator(nominal_indices, **kwargs)
                for estimator in self.all_estimators]

    def get_flow_mapping(self):
        return self.estimator_mapping
