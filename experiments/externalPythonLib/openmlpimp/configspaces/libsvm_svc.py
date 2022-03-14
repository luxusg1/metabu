from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant


def get_hyperparameter_search_space(seed):

    imputation = CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])

    C = UniformFloatHyperparameter("classifier__C", 0.03125, 32768, log=True, default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name="classifier__kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")
    degree = UniformIntegerHyperparameter("classifier__degree", 1, 5, default_value=3)
    gamma = UniformFloatHyperparameter("classifier__gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    coef0 = UniformFloatHyperparameter("classifier__coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter("classifier__shrinking", [True, False], default_value=True)
    tol = UniformFloatHyperparameter("classifier__tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = Constant("classifier__max_iter", -1)

    cs = ConfigurationSpace('sklearn.svm.SVC', seed)
    cs.add_hyperparameters([imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

    '''degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
    coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)
'''
    return cs

