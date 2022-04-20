import ConfigSpace


def get_hyperparameter_search_space(seed=None):
    """
    Text classification as defined by:
        M. J. Ferreira, P. Brazdil, Workflow Recommendation for Text Classification with Active Testing Method
    Note that this search space is defined in R, so we can not reinstantiate it in Python

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    representation = ConfigSpace.CategoricalHyperparameter(
        name='representation', choices=['Td-idf', 'freq'])
    stopword_remover = ConfigSpace.CategoricalHyperparameter(
        name='stopword_remover', choices=['none', 'smart', 'default'])
    stemmer = ConfigSpace.CategoricalHyperparameter(
        name='stemmer', choices=['none', 'porter'])
    sparsity = ConfigSpace.CategoricalHyperparameter(
        name='sparsity', choices=[0.98, 0.99])
    feature_selection = ConfigSpace.CategoricalHyperparameter(
        name='feature_selection', choices=['none', '> 0'])
    algorithm = ConfigSpace.CategoricalHyperparameter(
        name='algorithm', choices=['knn', 'ranger', 'JRip', 'svmLinear2', 'J48', 'nnet', 'lda', 'C5.0Tree'])

    cs.add_hyperparameters([
        representation,
        stopword_remover,
        stemmer,
        sparsity,
        feature_selection,
        algorithm
    ])

    return cs
