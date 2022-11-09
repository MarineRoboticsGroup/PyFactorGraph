import attr
from typing import List, Tuple, Dict, Optional
import numpy as np
from attrs import define, field
from py_factor_graph.utils.attrib_utils import (
    make_variable_name_validator,
    float_tuple_validator,
    positive_float_validator,
)
from py_factor_graph.utils.matrix_utils import get_rotation_matrix_from_theta


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
    translation_precision: float = attr.ib(validator=positive_float_validator)
    rotation_precision: float = attr.ib(validator=positive_float_validator)
    timestamp: Optional[float] = attr.ib(default=None)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def translation_vector(self):
        return np.array([self.x, self.y])

    @property
    def rotation_matrix(self):
        return get_rotation_matrix_from_theta(self.theta)

    @property
    def covariance(self):
        return np.diag(
            [
                1 / self.translation_precision,
                1 / self.translation_precision,
                1 / self.rotation_precision,
            ]
        )


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
    translation_precision: float = field(validator=positive_float_validator)

    @property
    def translation_vector(self):
        return np.array(self.position)

    @property
    def covariance(self):
        return np.diag([1 / self.translation_precision] * 2)
