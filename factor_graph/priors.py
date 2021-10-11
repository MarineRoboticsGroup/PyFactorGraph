import attr
from typing import List, Tuple, Dict, Optional
import numpy as np


@attr.s(frozen=True)
class PosePrior:
    """A prior on the robot pose

    Arguments:
        name (str): the name of the pose variable
        position (Tuple[float, float]): the prior of the position
        theta (float): the prior of the theta
        covariance (np.ndarray): the covariance of the prior
    """

    name: str = attr.ib()
    position: Tuple[float, float] = attr.ib()
    theta: float = attr.ib()
    covariance: np.ndarray = attr.ib()

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]


@attr.s(frozen=True)
class LandmarkPrior:
    """A prior on the landmark

    Arguments:
        name (str): the name of the landmark variable
        position (Tuple[float, float]): the prior of the position
        covariance (np.ndarray): the covariance of the prior
    """

    _name: str = attr.ib()
    _position: Tuple[float, float] = attr.ib()
    _covariance: np.ndarray = attr.ib()
