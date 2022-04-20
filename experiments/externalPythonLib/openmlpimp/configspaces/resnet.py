import ConfigSpace


def get_hyperparameter_search_space(seed=None):
    """
    Neural Network search space based on a best effort using the resnet
    implementation.

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
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[8, 16, 32, 64, 128, 256, 512], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='learning_rate_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='horizontal_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='vertical_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs
