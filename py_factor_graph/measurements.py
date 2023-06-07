import attr
from typing import Optional, Tuple, List, Union
import numpy as np
from py_factor_graph.utils.attrib_utils import (
    positive_float_validator,
    is_dimension,
    make_variable_name_validator,
    make_rot_matrix_validator,
    optional_float_validator,
)
from py_factor_graph.utils.matrix_utils import (
    get_covariance_matrix_from_measurement_precisions,
    get_quat_from_rotation_matrix,
    get_info_matrix_from_measurement_precisions,
)


@attr.s(frozen=False)
class PoseMeasurement2D:
    """
    An pose measurement

    Args:
        base_pose (str): the pose which the measurement is in the reference frame of
        to_pose (str): the name of the pose the measurement is to
        x (float): the measured change in x coordinate
        y (float): the measured change in y coordinate
        theta (float): the measured change in theta
        covariance (np.ndarray): a 3x3 covariance matrix from the measurement model
        timestamp (float): seconds since epoch
    """

    base_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    to_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    x: float = attr.ib(validator=attr.validators.instance_of(float))
    y: float = attr.ib(validator=attr.validators.instance_of(float))
    theta: float = attr.ib(validator=attr.validators.instance_of(float))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    rotation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Get the transformation matrix
        """
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta), self.x],
                [np.sin(self.theta), np.cos(self.theta), self.y],
                [0, 0, 1],
            ]
        )

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement
        """
        return np.array([self.x, self.y])

    @property
    def covariance(self) -> np.ndarray:
        """
        Get the covariance matrix
        """
        return get_covariance_matrix_from_measurement_precisions(
            self.translation_precision, self.rotation_precision, mat_dim=3
        )


@attr.s(frozen=False)
class PoseToLandmarkMeasurement2D:
    pose_name: str = attr.ib(validator=make_variable_name_validator("pose"))
    landmark_name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    x: float = attr.ib(validator=attr.validators.instance_of(float))
    y: float = attr.ib(validator=attr.validators.instance_of(float))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement
        """
        return np.array([self.x, self.y])

    @property
    def covariance(self) -> np.ndarray:
        """
        Get the covariance matrix
        """
        return np.diag([1 / self.translation_precision] * 2)


@attr.s(frozen=False)
class PoseToLandmarkMeasurement3D:
    pose_name: str = attr.ib(validator=make_variable_name_validator("pose"))
    landmark_name: str = attr.ib(validator=make_variable_name_validator("landmark"))
    x: float = attr.ib(validator=attr.validators.instance_of(float))
    y: float = attr.ib(validator=attr.validators.instance_of(float))
    z: float = attr.ib(validator=attr.validators.instance_of(float))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement
        """
        return np.array([self.x, self.y, self.z])

    @property
    def covariance(self) -> np.ndarray:
        """
        Get the covariance matrix
        """
        return np.diag([1 / self.translation_precision] * 3)


@attr.s(frozen=False)
class PoseMeasurement3D:
    """
    An pose measurement

    Args:
        base_pose (str): the pose which the measurement is in the reference frame of
        to_pose (str): the name of the pose the measurement is to
        translation (np.ndarray): the measured change in x, y, z coordinates
        rotation (np.ndarray): the measured change in rotation
        translation_precision (float): the weight of the translation measurement
        rotation_precision (float): the weight of the rotation measurement
        timestamp (float): seconds since epoch
    """

    base_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    to_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    translation: np.ndarray = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    rotation: np.ndarray = attr.ib(validator=make_rot_matrix_validator(3))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    rotation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix for the measurement

        Returns:
            np.ndarray: the 3x3 rotation matrix
        """
        return self.rotation

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Get the transformation matrix

        Returns:
            np.ndarray: the 4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    @property
    def translation_vector(self) -> np.ndarray:
        """
        Get the translation vector for the measurement

        Returns:
            np.ndarray: the 3x1 translation vector
        """
        return self.translation

    @property
    def x(self) -> float:
        """
        Get the x translation

        Returns:
            float: the x translation
        """
        return self.translation[0]

    @property
    def y(self) -> float:
        """
        Get the y translation

        Returns:
            float: the y translation
        """
        return self.translation[1]

    @property
    def z(self) -> float:
        """
        Get the z translation

        Returns:
            float: the z translation
        """
        return self.translation[2]

    @property
    def quat(self) -> np.ndarray:
        """
        Get the quaternion in the form [x, y, z, w]

        Returns:
            np.ndarray: the 4x1 quaternion
        """
        return get_quat_from_rotation_matrix(self.rotation)

    @property
    def covariance(self):
        """
        Get the 6x6 covariance matrix. Right now uses isotropic covariance
        for the translation and rotation respectively

        Returns:
            np.ndarray: the 6x6 covariance matrix
        """
        return get_covariance_matrix_from_measurement_precisions(
            self.translation_precision, self.rotation_precision, mat_dim=6
        )


@attr.s(frozen=True)
class AmbiguousPoseMeasurement2D:
    """
    An ambiguous odom measurement

    base_pose (str): the name of the base pose which the measurement is in the
        reference frame of
    measured_to_pose (str): the name of the pose the measurement thinks it is to
    true_to_pose (str): the name of the pose the measurement is to
    x (float): the change in x
    y (float): the change in y
    theta (float): the change in theta
    covariance (np.ndarray): a 3x3 covariance matrix
    timestamp (float): seconds since epoch
    """

    base_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    measured_to_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    true_to_pose: str = attr.ib(validator=make_variable_name_validator("pose"))
    x: float = attr.ib(validator=attr.validators.instance_of(float))
    y: float = attr.ib(validator=attr.validators.instance_of(float))
    theta: float = attr.ib(validator=attr.validators.instance_of(float))
    translation_precision: float = attr.ib(validator=positive_float_validator)
    rotation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(
        default=None, validator=optional_float_validator
    )

    @property
    def rotation_matrix(self):
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @property
    def transformation_matrix(self):
        """
        Get the transformation matrix
        """
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta), self.x],
                [np.sin(self.theta), np.cos(self.theta), self.y],
                [0, 0, 1],
            ]
        )

    @property
    def translation_vector(self):
        """
        Get the translation vector for the measurement
        """
        return np.array([self.x, self.y])

    @property
    def covariance(self):
        """
        Get the covariance matrix
        """
        return get_covariance_matrix_from_measurement_precisions(
            self.translation_precision, self.rotation_precision, mat_dim=3
        )


@attr.s(frozen=False)
class FGRangeMeasurement:
    """A range measurement

    Arguments:
        association (Tuple[str, str]): the data associations of the measurement.
            First association is the pose variable
        dist (float): The measured range
        stddev (float): The standard deviation
        timestamp (float): seconds since epoch
    """

    association: Tuple[str, str] = attr.ib()
    dist: float = attr.ib(validator=positive_float_validator)
    stddev: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(default=None)

    @association.validator
    def check_association(self, attribute, value: Tuple[str, str]):
        """Validates the association attribute

        Args:
            attribute ([type]): [description]
            value (Tuple[str, str]): the true_association attribute

        Raises:
            ValueError: is not a 2-tuple
            ValueError: the associations are identical
            ValueError: the first association is not a valid pose key
            ValueError: the second association is not a valid landmark key
        """
        assert all(isinstance(x, str) for x in value)
        if len(value) != 2:
            raise ValueError(
                "Range measurements must have exactly two variables associated with."
            )
        if value[0] == value[1]:
            raise ValueError(f"Range measurements must have unique variables{value}")
        if value[0].startswith("L"):
            raise ValueError("First association must be a pose - cannot start with L")
        if not value[1][0].isalpha() and value[1][1:].isnumeric():
            raise ValueError(f"Second association was not a valid variable: {value[1]}")

    @property
    def weight(self) -> float:
        """
        Get the weight of the measurement
        """
        return 1 / (self.stddev**2)

    @property
    def pose_key(self) -> str:
        """
        Get the pose key from the association
        """
        return self.association[0]

    @property
    def landmark_key(self) -> str:
        """
        Get the landmark key from the association
        """
        return self.association[1]

    @property
    def variance(self) -> float:
        """
        Get the variance of the measurement
        """
        return self.stddev**2

    @property
    def precision(self) -> float:
        """
        Get the precision of the measurement
        """
        return 1 / self.variance


@attr.s(frozen=True)
class AmbiguousFGRangeMeasurement:
    """A range measurement

    Arguments:
        var1 (str): one variable the measurement is associated with
        var2 (str): the other variable the measurement is associated with
        dist (float): The measured range
        stddev (float): The standard deviation
        timestamp (float): seconds since epoch
    """

    true_association: Tuple[str, str] = attr.ib()
    measured_association: Tuple[str, str] = attr.ib()
    dist: float = attr.ib()
    stddev: float = attr.ib()
    timestamp: Optional[float] = attr.ib(default=None)

    @true_association.validator
    def check_true_association(self, attribute, value: Tuple[str, str]):
        """Validates the true_association attribute

        Args:
            attribute ([type]): [description]
            value (Tuple[str, str]): the true_association attribute

        Raises:
            ValueError: is not a 2-tuple
            ValueError: the associations are identical
        """
        if len(value) != 2:
            raise ValueError(
                "Range measurements must have exactly two variables associated with."
            )
        if len(value) != len(set(value)):
            raise ValueError("Range measurements must have unique variables.")

    @measured_association.validator
    def check_measured_association(self, attribute, value):
        if len(value) != 2:
            raise ValueError(
                "Range measurements must have exactly two variables associated with."
            )
        if len(value) != len(set(value)):
            raise ValueError("Range measurements must have unique variables.")

    @property
    def weight(self):
        """
        Get the weight of the measurement
        """
        return 1 / (self.stddev**2)


POSE_MEASUREMENT_TYPES = Union[PoseMeasurement2D, PoseMeasurement3D]
POSE_TO_LANDMARK_MEASUREMENT_TYPES = Union[
    PoseToLandmarkMeasurement2D, PoseToLandmarkMeasurement3D
]
