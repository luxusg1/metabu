from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
import sklearn.tree


def get_hyperparameter_search_space(seed):
    cs = ConfigurationSpace('sklearn.ensemble.AdaBoostClassifier',
                            seed,
                            meta={"adaboostclassifier__base_estimator": sklearn.tree.DecisionTreeClassifier()})
    imputation = CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])
    n_estimators = UniformIntegerHyperparameter(
        name="classifier__n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = UniformFloatHyperparameter(
        name="classifier__learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = CategoricalHyperparameter(
        name="classifier__algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = UniformIntegerHyperparameter(
        name="classifier__base_estimator__max_depth", lower=1, upper=10, default_value=1, log=False)

    cs.add_hyperparameters([imputation, n_estimators, learning_rate, algorithm, max_depth])

    return cs
