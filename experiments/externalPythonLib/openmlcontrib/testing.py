import ConfigSpace
import unittest


class TestBase(unittest.TestCase):

    @staticmethod
    def _get_libsvm_svc_config_space():
        imputation = ConfigSpace.CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])

        C = ConfigSpace.UniformFloatHyperparameter("classifier__C", 0.03125, 32768, log=True, default_value=1.0)
        kernel = ConfigSpace.CategoricalHyperparameter(name="classifier__kernel",
                                                       choices=["rbf", "poly", "sigmoid"], default_value="rbf")
        degree = ConfigSpace.UniformIntegerHyperparameter("classifier__degree", 1, 5, default_value=3)
        gamma = ConfigSpace.UniformFloatHyperparameter("classifier__gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)

        coef0 = ConfigSpace.UniformFloatHyperparameter("classifier__coef0", -1, 1, default_value=0)
        shrinking = ConfigSpace.CategoricalHyperparameter("classifier__shrinking", [True, False], default_value=True)
        tol = ConfigSpace.UniformFloatHyperparameter("classifier__tol", 1e-5, 1e-1, default_value=1e-3, log=True)
        max_iter = ConfigSpace.UnParametrizedHyperparameter("classifier__max_iter", -1)

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

        degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
        coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs

    @staticmethod
    def _libsvm_expected_active_hyperparameters():
        expected_active_parameters = {
            'poly': {'classifier__C', 'classifier__kernel',
                     'classifier__degree', 'classifier__gamma',
                     'classifier__coef0', 'classifier__tol',
                     'classifier__shrinking', 'classifier__max_iter',
                     'imputation__strategy'},
            'rbf': {'classifier__C', 'classifier__kernel', 'classifier__gamma',
                    'classifier__tol', 'classifier__shrinking',
                    'classifier__max_iter', 'imputation__strategy'},
            'sigmoid': {'classifier__C', 'classifier__kernel',
                        'classifier__gamma', 'classifier__coef0',
                        'classifier__tol', 'classifier__shrinking',
                        'classifier__max_iter', 'imputation__strategy'},
        }

        return expected_active_parameters

    @staticmethod
    def _libsvm_expected_hyperparameter_types():
        return {
            'imputation__strategy': str,
            'classifier__C': float,
            'classifier__kernel': str,
            'classifier__degree': int,
            'classifier__gamma': float,
            'classifier__coef0': float,
            'classifier__shrinking': bool,
            'classifier__tol': float,
            'classifier__max_iter': int,
        }

    @staticmethod
    def _get_random_forest_default_search_space():
        cs = ConfigSpace.ConfigurationSpace()
        imputation = ConfigSpace.CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])
        n_estimators = ConfigSpace.Constant("classifier__n_estimators", 100)
        criterion = ConfigSpace.CategoricalHyperparameter("classifier__criterion", ["gini", "entropy"], default_value="gini")
        max_features = ConfigSpace.UniformFloatHyperparameter("classifier__max_features", 0., 1., default_value=0.5)

        # max_depth = ConfigSpace.UnParametrizedHyperparameter("classifier__max_depth", None)
        min_samples_split = ConfigSpace.UniformIntegerHyperparameter("classifier__min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter("classifier__min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = ConfigSpace.UnParametrizedHyperparameter("classifier__min_weight_fraction_leaf", 0.)
        # max_leaf_nodes = ConfigSpace.UnParametrizedHyperparameter("classifier__max_leaf_nodes", None)
        bootstrap = ConfigSpace.CategoricalHyperparameter("classifier__bootstrap", [True, False], default_value=True)
        cs.add_hyperparameters([imputation, n_estimators, criterion, max_features,
                                min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf,
                                bootstrap])
        return cs


__all__ = ['TestBase']
