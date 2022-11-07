import attr
from typing import List, Tuple, Dict, Optional
import numpy as np
from attrs import define, field
from py_factor_graph.utils.attrib_utils import (
    make_variable_name_validator,
    float_tuple_validator,
)


@define(frozen=True)
class PosePrior:
    """A prior on the robot pose

    Arguments:
        name (str): the name of the pose variable
        position (Tuple[float, float]): the prior of the position
        theta (float): the prior of the theta
        covariance (np.ndarray): the covariance of the prior
        timestamp (float): seconds since epoch
    """

    name: str = attr.ib()
    position: Tuple[float, float] = attr.ib()
    theta: float = attr.ib()
    covariance: np.ndarray = attr.ib()
    timestamp: Optional[float] = attr.ib(default=None)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]


@define(frozen=True)
class LandmarkPrior:
    """A prior on the landmark

    Arguments:
        name (str): the name of the landmark variable
        position (Tuple[float, float]): the prior of the position
        covariance (np.ndarray): the covariance of the prior
    """

    name: str = field(validator=make_variable_name_validator("landmark"))
    position: Tuple[float, float] = field(validator=float_tuple_validator)
    covariance: np.ndarray = field(validator=attr.validators.instance_of(np.ndarray))
