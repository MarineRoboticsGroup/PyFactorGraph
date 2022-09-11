from typing import List, Optional, Callable
import attr


def range_validator(instance, attribute, value):
    if value < 0:
        raise ValueError(f"Value {value} should not be negative")


def probability_validator(instance, attribute, value):
    """
    Return validator for probability.

    Args:
        value (float): value to validate

    Returns:
        None
    """
    if not isinstance(value, float):
        raise ValueError(f"{value} is not a float")
    if not 0 <= value <= 1:
        raise ValueError(f"Value {value} is not within range [0,1]")


def positive_float_validator(instance, attribute, value):
    """
    Return validator for positive float.

    Args:
        value (float): value to validate

    Returns:
        None
    """
    if not isinstance(value, float):
        raise ValueError(f"{value} is not a float")
    if value < 0:
        raise ValueError(f"Value {value} is not positive")


def positive_int_validator(instance, attribute, value) -> None:
    """
    Return validator for positive int.

    Args:
        value (int): value to validate

    Returns:
        None
    """
    if not isinstance(value, int):
        raise ValueError(f"{value} is not an int")
    if value < 0:
        raise ValueError(f"Value {value} is not positive")


def positive_int_tuple_validator(instance, attribute, value) -> None:
    """
    Return validator for positive int.

    Args:
        value (int): value to validate

    Returns:
        None
    """
    if not isinstance(value, tuple):
        raise ValueError(f"{value} is not a tuple")
    if not all(isinstance(x, int) for x in value):
        raise ValueError(f"At least one value in {value} is not an int")
    if not all(x >= 0 for x in value):
        raise ValueError(f"At least one value in {value} is negative")
