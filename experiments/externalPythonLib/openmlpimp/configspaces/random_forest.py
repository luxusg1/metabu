from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
UnParametrizedHyperparameter, Constant


def get_hyperparameter_search_space(seed):
    cs = ConfigurationSpace('sklearn.ensemble.RandomForestClassifier', seed)
    imputation = CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])
    n_estimators = Constant("classifier__n_estimators", 100)
    criterion = CategoricalHyperparameter(
        name = "classifier__criterion", choices = ["gini", "entropy"], default_value="gini")

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = UniformFloatHyperparameter(
        "classifier__max_features", lower=0., upper=1., default_value=0.5)

    # max_depth = UnParametrizedHyperparameter("classifier__max_depth", None)
    min_samples_split = UniformIntegerHyperparameter(
        "classifier__min_samples_split", lower=2, upper=20, default_value=2,log=False)
    min_samples_leaf = UniformIntegerHyperparameter(
        "classifier__min_samples_leaf", lower=1, upper=20, default_value=1,log=False)
    min_weight_fraction_leaf = UnParametrizedHyperparameter("classifier__min_weight_fraction_leaf", 0.)
    # max_leaf_nodes = UnParametrizedHyperparameter("classifier__max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter('classifier__min_impurity_decrease', 0.0)
    bootstrap = CategoricalHyperparameter(
        name = "classifier__bootstrap", choices = ["True", "False"], default_value="True")
    cs.add_hyperparameters([imputation, criterion, max_features,
                            min_samples_split, min_samples_leaf, bootstrap])

    return cs