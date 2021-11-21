import attr
from typing import Tuple, Optional
import numpy as np


@attr.s(frozen=True)
class PoseVariable:
    """A variable which is a robot pose

    Args:
        name (str): the name of the variable (defines the frame)
        true_position (Tuple[float, float]): the true position of the robot
        true_theta (float): the true orientation of the robot
        timestamp (float): seconds since epoch
    """

    name: str = attr.ib()
    true_position: Tuple[float, float] = attr.ib()
    true_theta: float = attr.ib()
    timestamp: Optional[float] = attr.ib(default=None)

    @property
    def rotation_matrix(self):
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
    def true_x(self):
        return self.true_position[0]

    @property
    def true_y(self):
        return self.true_position[1]


@attr.s(frozen=True)
class LandmarkVariable:
    """A variable which is a landmark

    Arguments:
        name (str): the name of the variable
        true_position (Tuple[float, float]): the true position of the landmark
    """

    name: str = attr.ib()
    true_position: Tuple[float, float] = attr.ib()

    @property
    def true_x(self):
        return self.true_position[0]

    @property
    def true_y(self):
        return self.true_position[1]
