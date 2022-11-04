from typing import List, Optional, Callable
import attr
import numpy as np


def is_dimension(instance, attribute, value) -> None:
    """
    Return validator for dimension.

    Args:
        value (int): value to validate

    Returns:
        None
    """
    if not isinstance(value, int):
        raise ValueError(f"{value} is not an int")
    if not value in [2, 3]:
        raise ValueError(f"Value {value} is not 2 or 3")


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
    if not isinstance(value, float) and not isinstance(value, int):
        raise ValueError(f"{value} is not a float (ints are also accepted)")
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


def make_rot_matrix_validator(dimension: int) -> Callable:
    """
    Return validator for rotation matrix.

    Args:
        value (np.ndarray): value to validate

    Returns:
        None
    """

    def rot_matrix_validator(instance, attribute, value) -> None:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{value} is not a np.ndarray")

        if not value.shape[0] == value.shape[1]:
            raise ValueError(f"Rotation matrix is not square {value.shape}")

        if not value.shape[0] == dimension:
            raise ValueError(f"Rotation matrix is not {dimension}x{dimension}")

        if not np.allclose(value @ value.T, np.eye(dimension)):
            raise ValueError(f"{value} is not orthogonal")

        if not np.allclose(np.linalg.det(value), 1):
            raise ValueError(f"{value} has determinant {np.linalg.det(value)}")

    return rot_matrix_validator


def rot_matrix_validator(instance, attribute, value) -> None:
    """
    Return validator for rotation matrix.

    Args:
        value (np.ndarray): value to validate

    Returns:
        None
    """
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{value} is not a np.ndarray")

    if not value.shape[0] == value.shape[1]:
        raise ValueError(f"Rotation matrix is not square {value.shape}")

    dim = value.shape[0]
    if not np.allclose(value @ value.T, np.eye(dim)):
        raise ValueError(f"{value} is not orthogonal")

    if not np.allclose(np.linalg.det(value), 1):
        raise ValueError(f"{value} has determinant {np.linalg.det(value)}")


def optional_float_validator(instance, attribute, value) -> None:
    """
    Return validator for optional float.

    Args:
        value (float): value to validate

    Returns:
        None
    """
    if value is not None:
        if not isinstance(value, float):
            raise ValueError(f"{value} is not a float")


def make_variable_name_validator(type: str) -> Callable:
    """
    Return validator for either pose or landmark names. Should be a string of
    form

    "<Letter><Number>"
    Poses should not start with 'L' : Poses = "A1", "B27", "C19"
    Landmarks should start with 'L' : Landmarks = "L1", "L2", "L3"

    Args:
        value (str): value to validate

    Returns:
        Callable: validator
    """
    valid_types = ["pose", "landmark"]
    assert (
        type in valid_types
    ), f"Type {type} is not valid, should be one of {valid_types}"

    def variable_name_validator(instance, attribute, value) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{value} is not a string")

        first_char = value[0]
        if type == "pose" and first_char == "L":
            raise ValueError(f"{value} starts with L - reserved for landmarks")
        elif type == "landmark" and first_char != "L":
            raise ValueError(
                f"{value} does not start with L - landmarks should start with L"
            )

        if not first_char.isalpha() or first_char.islower():
            raise ValueError(f"{value} does not start with a capital letter")

        if not value[1:].isdigit():
            raise ValueError(f"{value} does not end with a number")

    return variable_name_validator


def general_variable_name_validator(instance, attribute, value) -> None:
    """
    Return validator for variable names. Should be a string of form

    "<Letter><Number>"
    Poses should not start with 'L' : Poses = "A1", "B27", "C19"
    Landmarks should start with 'L' : Landmarks = "L1", "L2", "L3"

    Args:
        value (str): value to validate

    Returns:
        Callable: validator
    """
    if not isinstance(value, str):
        raise ValueError(f"{value} is not a string")

    first_char = value[0]
    if not first_char.isalpha() or first_char.islower():
        raise ValueError(f"{value} does not start with a capital letter")

    if not value[1:].isdigit():
        raise ValueError(f"{value} does not end with a number")
