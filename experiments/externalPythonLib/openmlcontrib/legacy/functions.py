import ConfigSpace
import numpy as np
import typing


def is_integer_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Checks whether hyperparameter is one of the following: Integer hyperparameter,
    Constant Hyperparameter with integer value, Unparameterized Hyperparameter
    with integer value or CategoricalHyperparameter with only integer options.

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    bool
        True iff the hyperparameter complies with the definition above, false
        otherwise
    """
    if isinstance(hyperparameter, ConfigSpace.hyperparameters.IntegerHyperparameter):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
            and isinstance(hyperparameter.value, int):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
            and isinstance(hyperparameter.value, int):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter) \
            and np.all([isinstance(choice, int) for choice in hyperparameter.choices]):
        return True
    return False


def is_boolean_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Checks whether hyperparameter is one of the following: Categorical
    hyperparameter with only boolean values, Constant Hyperparameter with
    boolean value or Unparameterized Hyperparameter with boolean value

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    bool
        True iff the hyperparameter complies with the definition above, false
        otherwise
    """
    if isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter) \
            and np.all([isinstance(choice, bool) for choice in hyperparameter.choices]):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
            and isinstance(hyperparameter.value, bool):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
            and isinstance(hyperparameter.value, bool):
        return True
    return False


def is_float_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Checks whether hyperparameter is one of the following: Float hyperparameter,
    Constant Hyperparameter with float value, Unparameterized Hyperparameter
    with float value or CategoricalHyperparameter with only integer options.

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    bool
        True iff the hyperparameter complies with the definition above, false
        otherwise
    """
    if isinstance(hyperparameter, ConfigSpace.hyperparameters.FloatHyperparameter):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
            and isinstance(hyperparameter.value, float):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
            and isinstance(hyperparameter.value, float):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter) \
            and np.all([isinstance(choice, float) for choice in hyperparameter.choices]):
        return True
    return False


def is_string_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Checks whether hyperparameter is one of the following: Categorical
    hyperparameter with only string values, Constant Hyperparameter with
    string value or Unparameterized Hyperparameter with string value

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    bool
        True iff the hyperparameter complies with the definition above, false
        otherwise
    """
    if isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter) \
            and np.all([isinstance(choice, str) for choice in hyperparameter.choices]):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
            and isinstance(hyperparameter.value, str):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
            and isinstance(hyperparameter.value, str):
        return True
    return False


def get_hyperparameter_datatype(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> typing.Callable:
    """
    Identifies and returns the datatype that a hyperparameter adhires to.
    TODO: Mixed types are currently badly supported.

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    Callable
        A Callable to cast the hyperparameter to the correct type
    """
    if is_boolean_hyperparameter(hyperparameter):
        return bool
    elif is_integer_hyperparameter(hyperparameter):
        return int
    elif is_float_hyperparameter(hyperparameter):
        return float
    elif is_string_hyperparameter(hyperparameter):
        return str
    else:
        raise ValueError('Hyperparameter type not determined yet. Please extend'
                         'this function. Hyperparameter: %s' % hyperparameter.name)
