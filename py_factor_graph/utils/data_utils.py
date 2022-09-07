import numpy as np
from typing import List


def get_covariance_matrix_from_list(covar_list: List) -> np.ndarray:
    """
    Converts a list of floats to a covariance matrix.

    Args:
        covar_list (List): a list of floats representing the covariance matrix

    Returns:
        np.ndarray: the covariance matrix
    """
    assert len(covar_list) == 3 * 3, f"{len(covar_list)} != 3x3"
    assert all(isinstance(val, float) for val in covar_list)
    covar_matrix = np.array(
        [
            [covar_list[0], covar_list[1], covar_list[2]],
            [covar_list[3], covar_list[4], covar_list[5]],
            [covar_list[6], covar_list[7], covar_list[8]],
        ]
    )

    assert np.allclose(
        covar_matrix, covar_matrix.T
    ), "Covariance matrix must be symmetric"
    assert covar_matrix.shape == (3, 3), "Covariance matrix must be 3x3"

    return covar_matrix


def get_theta_from_rotation_matrix(mat: np.ndarray) -> float:
    """ """
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def get_theta_from_transformation_matrix(mat: np.ndarray) -> float:
    """ """
    rotation_matrix = mat[0:2][0:2]
    return get_theta_from_rotation_matrix(rotation_matrix)
