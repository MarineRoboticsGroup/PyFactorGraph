import attr
from typing import Tuple
import numpy as np


@attr.s(frozen=True)
class PoseMeasurement:
    """
    An pose measurement

    base_pose (str): the name of the base pose which the measurement is in the
        reference frame of
    local_pose (str): the name of the pose the measurement is to
    x (float): the change in x coordinate of the measurement
    y (float): the change in y coordinate of the measurement
    theta (float): the change in theta
    covariance (np.ndarray): a 3x3 covariance matrix
    """

    base_pose: str = attr.ib()
    to_pose: str = attr.ib()
    x: float = attr.ib()
    y: float = attr.ib()
    theta: float = attr.ib()
    translation_weight: float = attr.ib()
    rotation_weight: float = attr.ib()

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
    def base_pose_idx(self) -> int:
        """
        Get the index of the base pose
        """
        return int(self.base_pose[1:])

    @property
    def to_pose_idx(self) -> int:
        """
        Get the index of the to pose
        """
        return int(self.to_pose[1:])

    @property
    def covariance(self):
        """
        Get the covariance matrix
        """
        return np.array(
            [
                [1 / self.translation_weight, 0, 0],
                [0, 1 / self.translation_weight, 0],
                [0, 0, 1 / self.rotation_weight],
            ]
        )


@attr.s(frozen=True)
class AmbiguousPoseMeasurement:
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
    """

    base_pose: str = attr.ib()
    measured_to_pose: str = attr.ib()
    true_to_pose: str = attr.ib()
    x: float = attr.ib()
    y: float = attr.ib()
    theta: float = attr.ib()
    translation_weight: float = attr.ib()
    rotation_weight: float = attr.ib()

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
        return np.array(
            [
                [1 / self.translation_weight, 0, 0],
                [0, 1 / self.translation_weight, 0],
                [0, 0, 1 / self.rotation_weight],
            ]
        )


@attr.s(frozen=True)
class FGRangeMeasurement:
    """A range measurement

    Arguments:
        association (Tuple[str]): the data associations of the measurement
        dist (float): The measured range
        stddev (float): The standard deviation
    """

    association: Tuple[str, str] = attr.ib()
    dist: float = attr.ib()
    stddev: float = attr.ib()

    @association.validator
    def check_association(self, attribute, value):
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
        return 1 / (self.stddev ** 2)

    @property
    def pose_idx(self):
        """
        Get the index of the pose
        """
        return int(self.var1[1:])

    @property
    def landmark_idx(self):
        """
        Get the index of the landmark
        """
        return int(self.var2[1:])


@attr.s(frozen=True)
class AmbiguousFGRangeMeasurement:
    """A range measurement

    Arguments:
        var1 (str): one variable the measurement is associated with
        var2 (str): the other variable the measurement is associated with
        dist (float): The measured range
        stddev (float): The standard deviation
    """

    true_association: Tuple[str, str] = attr.ib()
    measured_association: Tuple[str, str] = attr.ib()
    dist: float = attr.ib()
    stddev: float = attr.ib()

    @true_association.validator
    def check_true_association(self, attribute, value):
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
        return 1 / (self.stddev ** 2)

    @property
    def pose_idx(self):
        """
        Get the index of the pose
        """
        return int(self.var1[1:])

    @property
    def landmark_idx(self):
        """
        Get the index of the landmark
        """
        return int(self.var2[1:])
