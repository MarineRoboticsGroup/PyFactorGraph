import attr
from typing import Tuple, Optional, Union
import numpy as np
from py_factor_graph.utils.matrix_utils import get_quat_from_rotation_matrix
from py_factor_graph.utils.attrib_utils import (
    optional_float_validator,
    make_rot_matrix_validator,
    make_variable_name_validator,
)


@attr.s(frozen=True)
class PoseVariable2D:
    """A variable which is a robot pose

    Args:
        name (str): the name of the variable (defines the frame)
        true_position (Tuple[float, float]): the true position of the robot
        true_theta (float): the true orientation of the robot
        timestamp (float): seconds since epoch
    """

    name: str = attr.ib(validator=make_variable_name_validator("pose"))
    true_position: Tuple[float, float] = attr.ib()
    true_theta: float = attr.ib(validator=attr.validators.instance_of(float))
    timestamp: Optional[float] = attr.ib(default=None)

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 2:
            raise ValueError(f"true_position should be a tuple of length 2")
        assert all(isinstance(x, float) for x in value)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.true_theta), -np.sin(self.true_theta)],
                [np.sin(self.true_theta), np.cos(self.true_theta)],
            ]
        )

    @property
    def position_vector(self) -> np.ndarray:
        """
        Get the position vector for the measurement
        """
        return np.array(self.true_position)

    @property
    def true_x(self) -> float:
        return self.true_position[0]

    @property
    def true_y(self) -> float:
        return self.true_position[1]

    @property
    def true_z(self) -> float:
        return 0

    @property
    def true_quat(self) -> np.ndarray:
        quat = np.array(
            [0.0, 0.0, np.sin(self.true_theta / 2), np.cos(self.true_theta / 2)]
        )
        return quat

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            np.ndarray: the transformation matrix
        """
        T = np.eye(3)
        T[0:2, 0:2] = self.rotation_matrix
        T[0, 2] = self.true_x
        T[1, 2] = self.true_y
        return T


@attr.s(frozen=True)
class PoseVariable3D:
    """A variable which is a robot pose

    Args:
        name (str): the name of the variable (defines the frame)
        true_position (Tuple[float, float, float]): the true position of the robot
        true_rotation (np.ndarray): the true orientation of the robot
        timestamp (float): seconds since epoch
    """

    name: str = attr.ib(validator=make_variable_name_validator("pose"))
    true_position: Tuple[float, float, float] = attr.ib()
    true_rotation: np.ndarray = attr.ib(validator=make_rot_matrix_validator(3))
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 3:
            raise ValueError(f"true_position should be a tuple of length 3: {value}")
        assert all(isinstance(x, float) for x in value)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement
        """
        return self.true_rotation

    @property
    def position_vector(self) -> np.ndarray:
        """
        Get the position vector for the measurement
        """
        return np.array(self.true_position)

    @property
    def true_x(self) -> float:
        return self.true_position[0]

    @property
    def true_y(self) -> float:
        return self.true_position[1]

    @property
    def true_z(self) -> float:
        return self.true_position[2]

    @property
    def true_quat(self) -> np.ndarray:
        rot = self.rotation_matrix
        quat = get_quat_from_rotation_matrix(rot)
        return quat

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Returns the transformation matrix representing the true latent pose
        of this variable

        Returns:
            np.ndarray: the transformation matrix
        """
        T = np.eye(self.dimension + 1)
        T[: self.dimension, : self.dimension] = self.true_rotation
        T[: self.dimension, self.dimension] = self.true_position
        assert T.shape == (4, 4)
        return T


@attr.s(frozen=True)
class LandmarkVariable2D:
    """A variable which is a landmark

    Arguments:
        name (str): the name of the variable
        true_position (Tuple[float, float]): the true position of the landmark
    """

    name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    true_position: Tuple[float, float] = attr.ib()

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 2:
            raise ValueError(f"true_position should be a tuple of length 2")
        assert all(isinstance(x, float) for x in value)

    @property
    def true_x(self):
        return self.true_position[0]

    @property
    def true_y(self):
        return self.true_position[1]


@attr.s(frozen=True)
class LandmarkVariable3D:
    """A variable which is a landmark

    Arguments:
        name (str): the name of the variable
        true_position (Tuple[float, float, float]): the true position of the landmark
    """

    name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    true_position: Tuple[float, float, float] = attr.ib()

    @true_position.validator
    def _check_true_position(self, attribute, value):
        if len(value) != 3:
            raise ValueError(f"true_position should be a tuple of length 3")
        assert all(isinstance(x, float) for x in value)

    @property
    def true_x(self):
        return self.true_position[0]

    @property
    def true_y(self):
        return self.true_position[1]

    @property
    def true_z(self):
        return self.true_position[2]


POSE_VARIABLE_TYPES = Union[PoseVariable2D, PoseVariable3D]
LANDMARK_VARIABLE_TYPES = Union[LandmarkVariable2D, LandmarkVariable3D]
