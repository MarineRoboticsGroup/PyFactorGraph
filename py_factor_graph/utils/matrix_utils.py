import numpy as np
import scipy.linalg as la  # type: ignore
import scipy.spatial
from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


def get_matrix_determinant(mat: np.ndarray) -> float:
    """returns the determinant of the matrix

    Args:
        mat (np.ndarray): [description]

    Returns:
        float: [description]
    """
    _check_square(mat)
    return float(np.linalg.det(mat))


def round_to_special_orthogonal(mat: np.ndarray) -> np.ndarray:
    """
    Rounds a matrix to special orthogonal form.

    Args:
        mat (np.ndarray): the matrix to round

    Returns:
        np.ndarray: the rounded matrix
    """
    _check_square(mat)
    _check_rotation_matrix(mat, assert_test=False)
    S, D, Vh = la.svd(mat)
    R_so = S @ Vh
    if np.linalg.det(R_so) < 0:
        Vh[-1, :] *= -1
        R_so = S @ Vh
    _check_rotation_matrix(R_so, assert_test=True)
    return R_so


def get_measurement_precisions_from_info_matrix(
    info_mat: np.ndarray, matrix_dim: Optional[int] = None
) -> Tuple[float, float]:
    """Computes the optimal measurement precisions from the information matrix

    Based on SE-Sync:SESync_utils.cpp:113-124

    Args:
        info_mat (np.ndarray): the information matrix

    Returns:
        Tuple[float, float]: (translation precision, rotation precision)
    """
    _check_square(info_mat)
    dim = info_mat.shape[0]
    if matrix_dim is not None:
        assert (
            matrix_dim == dim
        ), f"matrix_dim {matrix_dim} must match info_mat dim {dim}"

    assert dim in [3, 6], "information matrix must be 3x3 or 6x6"
    covar_mat = la.inv(info_mat)
    trans_precision, rot_precision = get_measurement_precisions_from_covariance_matrix(
        covar_mat, matrix_dim
    )
    return (trans_precision, rot_precision)


def get_measurement_precisions_from_covariance_matrix(
    covar_mat: np.ndarray, matrix_dim: Optional[int] = None
) -> Tuple[float, float]:
    """Computes the optimal measurement precisions from the covariance matrix

    Based on SE-Sync:SESync_utils.cpp:113-124

    Args:
        covar_mat (np.ndarray): the covariance matrix

    Returns:
        Tuple[float, float]: (translation precision, rotation precision)
    """
    _check_square(covar_mat)
    dim = covar_mat.shape[0]
    if matrix_dim is not None:
        assert (
            matrix_dim == dim
        ), f"matrix_dim {matrix_dim} must match info_mat dim {dim}"

    assert dim in [3, 6], "information matrix must be 3x3 or 6x6"

    def _get_trans_precision() -> float:
        if dim == 3:
            trans_covar = covar_mat[:2, :2]
            trans_precision = 2 / (np.trace(trans_covar))
        elif dim == 6:
            trans_covar = covar_mat[:3, :3]
            trans_precision = 3 / (np.trace(trans_covar))
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        return trans_precision

    def _get_rot_precision() -> float:
        if dim == 3:
            rot_precision = 1 / covar_mat[2, 2]
        elif dim == 6:
            rot_cov = covar_mat[3:, 3:]
            rot_precision = 3 / (2 * np.trace(rot_cov))
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        return rot_precision

    trans_precision = _get_trans_precision()
    rot_precision = _get_rot_precision()
    return trans_precision, rot_precision


def get_measurement_precisions_from_covariances(
    trans_cov: float, rot_cov: float
) -> Tuple[float, float]:
    """Converts the trans covariance and rot covariance to measurement
    precisions (assuming isotropic noise)

    Args:
        trans_cov (float): covariance of translation measurements
        rot_cov (float): covariance of rotation measurements

    Returns:
        Tuple[float, float]: (trans precision, rot precision)
    """
    trans_precision = 1 / trans_cov
    rot_precision = 1 / (2 * rot_cov)
    return trans_precision, rot_precision


def get_info_matrix_from_measurement_precisions(
    trans_precision: float, rot_precision: float, mat_dim: int
) -> np.ndarray:
    assert mat_dim in [3, 6], f"Only support 3x3 or 6x6 info matrices"
    if mat_dim == 3:
        trans_info = [trans_precision] * 2
        rot_info = [2 * rot_precision]
    if mat_dim == 6:
        trans_info = [trans_precision] * 3
        rot_info = [2 * rot_precision] * 3
    info_mat = np.diag(trans_info + rot_info)
    return info_mat


def get_covariance_matrix_from_measurement_precisions(
    trans_precision: float, rot_precision: float, mat_dim: int
) -> np.ndarray:
    info_mat = get_info_matrix_from_measurement_precisions(
        trans_precision, rot_precision, mat_dim
    )
    cov_mat = la.inv(info_mat)
    return cov_mat


def get_theta_from_rotation_matrix_so_projection(mat: np.ndarray) -> float:
    """
    Returns theta from the projection of the matrix M onto the special
    orthogonal group

    Args:
        mat (np.ndarray): the candidate rotation matrix

    Returns:
        float: theta

    """
    R_so = round_to_special_orthogonal(mat)
    return get_theta_from_rotation_matrix(R_so)


def get_theta_from_rotation_matrix(mat: np.ndarray) -> float:
    """
    Returns theta from a matrix M

    Args:
        mat (np.ndarray): the candidate rotation matrix

    Returns:
        float: theta
    """
    _check_square(mat)
    assert mat.shape == (2, 2)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def get_random_vector(dim: int) -> np.ndarray:
    """Returns a random vector of size dim

    Args:
        dim (int): the dimension of the vector

    Returns:
        np.ndarray: the random vector
    """
    return np.random.rand(dim)


def get_rotation_matrix_from_theta(theta: float) -> np.ndarray:
    """Returns the rotation matrix from theta

    Args:
        theta (float): the angle of rotation

    Returns:
        np.ndarray: the rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def get_rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix from roll, pitch, yaw angles

    Args:
        rpy (np.ndarray): the roll, pitch, yaw angles

    Returns:
        np.ndarray: the rotation matrix
    """
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    alpha = yaw
    beta = pitch
    gamma = roll
    m11 = np.cos(alpha) * np.cos(beta)
    m12 = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    m13 = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    m21 = np.sin(alpha) * np.cos(beta)
    m22 = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    m23 = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    m31 = -np.sin(beta)
    m32 = np.cos(beta) * np.sin(gamma)
    m33 = np.cos(beta) * np.cos(gamma)
    return np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])


def get_rotation_matrix_from_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """Returns the rotation matrix from the transformation matrix

    Args:
        T (np.ndarray): the transformation matrix

    Returns:
        np.ndarray: the rotation matrix
    """
    _check_square(T)
    dim = T.shape[0] - 1
    return T[:dim, :dim]


def get_rotation_matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    """Returns the rotation matrix from a quaternion

    Args:
        quat (np.ndarray): the quaternion

    Returns:
        np.ndarray: the rotation matrix
    """
    assert quat.shape == (4,)
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    assert isinstance(rot, scipy.spatial.transform.Rotation)

    # get as a matrix
    rot_mat = rot.as_matrix()
    assert isinstance(rot_mat, np.ndarray)
    assert rot_mat.shape == (3, 3)

    _check_rotation_matrix(rot_mat, assert_test=True)
    return rot_mat


def get_quat_from_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    """Returns the quaternion from a rotation matrix

    Args:
        mat (np.ndarray): the rotation matrix

    Returns:
        np.ndarray: the quaternion
    """
    _check_rotation_matrix(mat)
    mat_dim = mat.shape[0]

    if mat_dim == 2:
        rot_matrix = np.eye(3)
        rot_matrix[:2, :2] = mat
    else:
        rot_matrix = mat

    rot = scipy.spatial.transform.Rotation.from_matrix(rot_matrix)
    assert isinstance(rot, scipy.spatial.transform.Rotation)
    quat = rot.as_quat()
    assert isinstance(quat, np.ndarray)
    return quat


def get_theta_from_transformation_matrix(T: np.ndarray) -> float:
    """Returns the angle theta from a transformation matrix

    Args:
        T (np.ndarray): the transformation matrix

    Returns:
        float: the angle theta
    """
    assert T.shape == (3, 3), "transformation matrix must be 3x3"
    return get_theta_from_rotation_matrix(
        get_rotation_matrix_from_transformation_matrix(T)
    )


def get_translation_from_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """Returns the translation from a transformation matrix

    Args:
        T (np.ndarray): the transformation matrix

    Returns:
        np.ndarray: the translation
    """
    _check_square(T)
    dim = T.shape[0] - 1
    return T[:dim, dim]


def get_random_rotation_matrix(dim: int = 2) -> np.ndarray:
    """Returns a random rotation matrix of size dim x dim"""
    if dim == 2:
        theta = 2 * np.pi * np.random.rand()
        return get_rotation_matrix_from_theta(theta)
    else:
        raise NotImplementedError("Only implemented for dim = 2")


def get_random_transformation_matrix(dim: int = 2) -> np.ndarray:
    if dim == 2:
        R = get_random_rotation_matrix(dim)
        t = get_random_vector(dim)
        return make_transformation_matrix(R, t)
    else:
        raise NotImplementedError("Only implemented for dim = 2")


def make_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Returns the transformation matrix from a rotation matrix and translation vector

    Args:
        R (np.ndarray): the rotation matrix
        t (np.ndarray): the translation vector

    Returns:
        np.ndarray: the transformation matrix
    """
    _check_rotation_matrix(R)
    assert len(t) == len(
        R
    ), f"Rotation and translations have different dimensions: {len(t)} != {len(R)}"
    dim = len(t)
    T = np.eye(dim + 1)
    T[:dim, :dim] = R
    T[:dim, dim] = t
    _check_transformation_matrix(T)
    return T


def make_transformation_matrix_from_theta(
    theta: float,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Returns the transformation matrix from theta and translation

    Args:
        theta (float): the angle of rotation
        translation (np.ndarray): the translation

    Returns:
        np.ndarray: the transformation matrix
    """
    R = get_rotation_matrix_from_theta(theta)
    return make_transformation_matrix(R, translation)


def make_transformation_matrix_from_rpy(
    rpy: np.ndarray, trans: np.ndarray
) -> np.ndarray:
    """
    Returns the transformation matrix from rpy

    Args:
        rpy (np.ndarray): the rpy vector

    Returns:
        np.ndarray: the transformation matrix
    """
    R = get_rotation_matrix_from_theta(rpy[0])
    return make_transformation_matrix(R, trans)


def get_relative_transform_between_poses(
    pose_0: np.ndarray, pose_1: np.ndarray
) -> np.ndarray:
    """Returns the relative transformation between two poses
    expressed in the same base frame

    Args:
        pose_0 (np.ndarray): the first pose
        pose_1 (np.ndarray): the second pose

    Raises:
        error: _description_
        ValueError: _description_
        ValueError: _description_
    """
    if pose_0.shape != pose_1.shape:
        raise ValueError(
            f"Poses have different shapes: {pose_0.shape} != {pose_1.shape}"
        )

    pose_0_inv = np.linalg.inv(pose_0)
    return np.dot(pose_0_inv, pose_1)


def get_relative_rot_and_trans_between_poses(
    pose_0: np.ndarray, pose_1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the relative rotation matrix and translation vector from pose_0
    to pose_1, assuming they are expressed in the same base frame

        Args:
            pose_0 (np.ndarray): the first pose
            pose_1 (np.ndarray): the second pose

        Returns:
            Tuple[np.ndarray, np.ndarray]: relative rotation matrix and translation vector
    """
    relative_trans = get_relative_transform_between_poses(pose_0, pose_1)
    _check_square(relative_trans)
    rot = get_rotation_matrix_from_transformation_matrix(relative_trans)
    trans = get_translation_from_transformation_matrix(relative_trans)
    return (rot, trans)


#### test functions ####


def _check_rotation_matrix(R: np.ndarray, assert_test: bool = False):
    """
    Checks that R is a rotation matrix.

    Args:
        R (np.ndarray): the candidate rotation matrix
        assert_test (bool): if false just print if not rotation matrix, otherwise raise error
    """
    d = R.shape[0]
    is_orthogonal = np.allclose(R @ R.T, np.eye(d), rtol=1e-3, atol=1e-3)
    if not is_orthogonal:
        # print(f"R not orthogonal: {R @ R.T}")
        if assert_test:
            raise ValueError(f"R is not orthogonal {R @ R.T}")

    has_correct_det = abs(np.linalg.det(R) - 1) < 1e-3
    if not has_correct_det:
        # print(f"R det != 1: {np.linalg.det(R)}")
        if assert_test:
            raise ValueError(f"R det incorrect {np.linalg.det(R)}")


def _check_square(mat: np.ndarray):
    assert mat.shape[0] == mat.shape[1], "matrix must be square"


def _check_symmetric(mat):
    assert np.allclose(mat, mat.T)


def _check_psd(mat: np.ndarray):
    """Checks that a matrix is positive semi-definite"""
    assert isinstance(mat, np.ndarray)
    if is_diagonal(mat):
        assert np.all(np.diag(mat) >= 0)
    else:
        min_eigval = np.min(la.eigvalsh(mat))
        if min_eigval < 0:
            logger.error(f"Matrix is not positive semi-definite:\n{mat}")
        assert min_eigval + 1e-1 >= 0.0, f"min eigenvalue is {min_eigval}"


def _check_is_laplacian(L: np.ndarray):
    """Checks that a matrix is a Laplacian based on well-known properties

    Must be:
        - symmetric
        - ones vector in null space of L
        - no negative eigenvalues

    Args:
        L (np.ndarray): the candidate Laplacian
    """
    assert isinstance(L, np.ndarray)
    _check_symmetric(L)
    _check_psd(L)
    ones = np.ones(L.shape[0])
    zeros = np.zeros(L.shape[0])
    assert np.allclose(L @ ones, zeros), f"L @ ones != zeros: {L @ ones}"


def _check_transformation_matrix(
    T: np.ndarray, assert_test: bool = True, dim: Optional[int] = None
):
    """Checks that the matrix passed in is a homogeneous transformation matrix.
    If assert_test is True, then this is in the form of assertions, otherwise we
    just print out error messages but continue

    Args:
        T (np.ndarray): the matrix to test
        assert_test (bool, optional): Whether this is a 'hard' test and is
        asserted or just a 'soft' test and only prints message if test fails. Defaults to True.
    """
    _check_square(T)
    matrix_dim = T.shape[0]
    if dim is not None:
        assert (
            matrix_dim == dim + 1
        ), f"matrix dimension {matrix_dim} != dim + 1 {dim + 1}"

    assert matrix_dim in [
        3,
        4,
    ], f"Was {T.shape} but must be 3x3 or 4x4 for a transformation matrix"

    # check that is rotation matrix in upper left block
    R = T[:-1, :-1]
    _check_rotation_matrix(R, assert_test=assert_test)

    # check that the bottom row is [0, 0, 1]
    bottom = T[-1, :]
    bottom_expected = np.array([0] * (matrix_dim - 1) + [1])
    assert np.allclose(
        bottom.flatten(), bottom_expected
    ), f"Transformation matrix bottom row is {bottom} but should be {bottom_expected}"


def is_diagonal(mat: np.ndarray) -> bool:
    """Checks if a matrix is diagonal

    Args:
        mat (np.ndarray): the matrix to check

    Returns:
        bool: True if diagonal, False otherwise
    """
    return np.allclose(mat, np.diag(np.diag(mat)))


def is_approx_isotropic(mat: np.ndarray, eps: float = 15e-1) -> bool:
    """Checks that the covariance/info matrix is approximately isotropic in the
    translation and rotation components

    Args:
        covar (np.ndarray): the covariance matrix
    """
    try:
        _check_psd(mat)
    except AssertionError:
        return False

    dim = mat.shape[0]
    assert dim in [
        3,
        6,
    ], f"matrix must be 3x3 or 6x6, received {mat.shape}"

    # compute the difference
    if dim == 3:
        cutoff_idx = 2
    elif dim == 6:
        cutoff_idx = 3

    # construct a diagonal matrix and check that the difference is small
    trans_covar = min(np.diag(mat[:cutoff_idx, :cutoff_idx]))
    rot_covar = min(np.diag(mat[cutoff_idx:, cutoff_idx:]))

    if dim == 3:
        diagonal = [trans_covar] * 2 + [rot_covar]
    elif dim == 6:
        diagonal = [trans_covar] * 3 + [rot_covar] * 3
    diag_mat = np.diag(diagonal)

    translations_close = np.allclose(
        diag_mat[:cutoff_idx, :cutoff_idx],
        mat[:cutoff_idx, :cutoff_idx],
        atol=trans_covar * eps,
    )

    rotations_close = np.allclose(
        diag_mat[cutoff_idx:, cutoff_idx:],
        mat[cutoff_idx:, cutoff_idx:],
        atol=rot_covar * eps,
    )

    return translations_close and rotations_close


#### print functions ####


def _print_eigvals(
    M: np.ndarray, name: str = None, print_eigvec: bool = False, symmetric: bool = True
):
    """print the eigenvalues of a matrix"""

    if name is not None:
        print(name)

    if print_eigvec:
        # get the eigenvalues of the matrix
        if symmetric:
            eigvals, eigvecs = la.eigh(M)
        else:
            eigvals, eigvecs = la.eig(M)

        # sort the eigenvalues and eigenvectors
        idx = eigvals.argsort()[::1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        print(f"eigenvectors: {eigvecs}")
    else:
        if symmetric:
            eigvals = la.eigvalsh(M)
        else:
            eigvals = la.eigvals(M)
        print(f"eigenvalues\n{eigvals}")

    print("\n\n\n")


def _matprint_block(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    num_col = mat.shape[1]
    row_spacer = ""
    for _ in range(num_col):
        row_spacer += "__ __ __ "
    for j, x in enumerate(mat):
        if j % 2 == 0:
            print(row_spacer)
            print("")
        for i, y in enumerate(x):
            if i % 2 == 1:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end=" | ")
            else:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")

    print(row_spacer)
    print("\n\n\n")


def apply_transformation_matrix_perturbation(
    transformation_matrix,
    perturb_magnitude: Optional[float],
    perturb_rotation: Optional[float],
) -> np.ndarray:
    """Applies a random SE(2) perturbation to a transformation matrix

    Args:
        transformation_matrix ([type]): [description]
        perturb_magnitude (Optional[float]): [description]
        perturb_rotation (Optional[float]): [description]

    Returns:
        np.ndarray: [description]
    """
    _check_transformation_matrix(transformation_matrix)

    # get the x/y perturbation
    perturb_direction = np.random.uniform(0, 2 * np.pi)
    perturb_x = np.cos(perturb_direction) * perturb_magnitude
    perturb_y = np.sin(perturb_direction) * perturb_magnitude

    # get the rotation perturbation
    perturb_theta = np.random.choice([-1, 1]) * perturb_rotation

    # compose the perturbation into a transformation matrix
    rand_trans = np.eye(3)
    rand_trans[:2, :2] = get_rotation_matrix_from_theta(perturb_theta)
    rand_trans[:2, 2] = perturb_x, perturb_y
    _check_transformation_matrix(rand_trans)

    # perturb curr pose
    return transformation_matrix @ rand_trans
