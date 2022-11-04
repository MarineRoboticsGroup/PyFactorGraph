from typing import Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
from attrs import define, field
import attrs
from os.path import expanduser, join

from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    POSE_VARIABLE_TYPES,
    LandmarkVariable2D,
    LandmarkVariable3D,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    POSE_MEASUREMENT_TYPES,
    FGRangeMeasurement,
)
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.name_utils import get_robot_char_from_number
from py_factor_graph.utils.attrib_utils import (
    make_variable_name_validator,
    general_variable_name_validator,
)
from py_factor_graph.utils.matrix_utils import (
    _check_transformation_matrix,
    get_random_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_translation_from_transformation_matrix,
)
from py_factor_graph.utils.attrib_utils import (
    positive_float_validator,
    positive_int_validator,
)
import random

import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)


@define
class RangePairCalibration:
    """A class for calibrating the range measurement between two robots or one robot and a landmark

    Attributes:
        var1 (str): The first pose of the first robot (e.g. "A0")
        var2 (str): The second robot or beacon (e.g. "B0" or "L2")
        bias (float): The bias of the range measurement.
        covariance (float): The covariance value.
    """

    var1: str = field(validator=make_variable_name_validator("pose"))
    var2: str = field(validator=general_variable_name_validator)
    bias: float = field(validator=attrs.validators.instance_of((float, int)))
    covariance: float = field(validator=positive_float_validator)
    num_measurements: int = field(validator=positive_int_validator)

    @property
    def stddev(self):
        return np.sqrt(self.covariance)

    def __str__(self):
        """Return a string representation of the calibration."""
        return f"{self.var1} - {self.var2} ({self.num_measurements} measures): {self.bias} +- {self.stddev} stddev"


def _get_range_pair_calibrations_from_bias_measurements(
    bias_list: List[float],
    var1: str,
    var2: str,
) -> RangePairCalibration:
    """Get the calibration from a list of bias measurements

    Args:
        bias_list (List[float]): A list of bias measurements
        var1 (str): The first variable name
        var2 (str): The second variable name

    Returns:
        RangePairCalibration: The calibration
    """
    bias = np.mean(bias_list).astype(float)
    covariance = np.var(bias_list).astype(float)
    num_measurements = len(bias_list)
    return RangePairCalibration(
        var1=var1,
        var2=var2,
        bias=bias,
        covariance=covariance,
        num_measurements=num_measurements,
    )


def get_range_measure_calibrations(
    fg: FactorGraphData,
) -> Dict[Tuple[str, str], RangePairCalibration]:
    """Get the range measurement calibration values for a factor graph. This
    assumes that the calibration is orientation agnostic and just consists of a
    constant bias term and a covariance.

    In other words, the sensor noise is treated as a constant gaussian
    distribution.

    measured_dist = true_dist + N(bias, covariance)

    where bias and covariance are unique for each device pairing (i.e.
    robot-robot or robot-beacon pair).

    Args:
        fg (FactorGraphData): The factor graph.

    Returns:
        Dict[Tuple[str, str], RangePairCalibration]: A dictionary of the calibration values.
    """

    var_true_positions_dict = fg.variable_true_positions_dict

    # get all possible robot-robot and robot-beacon pairs
    def _get_all_variable_pairs() -> List[Tuple[str, str]]:
        # only get the name of the first pose for each robot
        robot_variable_names = [pose_chain[0].name for pose_chain in fg.pose_variables]
        landmark_variable_names = [landmark.name for landmark in fg.landmark_variables]
        robot_robot_pairs = []
        for i in range(len(robot_variable_names)):
            for j in range(i + 1, len(robot_variable_names)):
                robot_robot_pairs.append(
                    (robot_variable_names[i], robot_variable_names[j])
                )

        robot_landmark_pairs = []
        for robot in robot_variable_names:
            for landmark in landmark_variable_names:
                robot_landmark_pairs.append((robot, landmark))

        variable_name_pairs = robot_robot_pairs + robot_landmark_pairs
        return variable_name_pairs

    def _get_true_distance_between_variables(var1: str, var2: str) -> float:
        true_position1 = np.array(var_true_positions_dict[var1])
        true_position2 = np.array(var_true_positions_dict[var2])
        true_dist = np.linalg.norm(true_position1 - true_position2).astype(float)
        return true_dist

    # for each robot-robot or robot-beacon pair, record the biases measured
    pair_to_all_biases_maps: Dict[Tuple[str, str], List[float]] = {
        pair: [] for pair in _get_all_variable_pairs()
    }

    # record the biases measured for each pair
    for range_measure in fg.range_measurements:
        var1, var2 = range_measure.association
        true_dist = _get_true_distance_between_variables(var1, var2)
        measured_dist = range_measure.dist
        measured_bias = measured_dist - true_dist

        pair1_var = var1[0] + "0"
        if var2.startswith("L"):
            pair2_var = var2
        else:
            pair2_var = var2[0] + "0"
        var_pair = (pair1_var, pair2_var)

        pair_to_all_biases_maps[var_pair].append(measured_bias)

    # get the calibration for each pair
    pair_to_calibration_map: Dict[Tuple[str, str], RangePairCalibration] = {
        pair: _get_range_pair_calibrations_from_bias_measurements(
            bias_list, pair[0], pair[1]
        )
        for pair, bias_list in pair_to_all_biases_maps.items()
    }

    return pair_to_calibration_map


if __name__ == "__main__":
    from py_factor_graph.parsing.parse_goats_data import (
        GoatsParser,
        get_data_and_beacon_files,
    )
    from pathlib import Path

    dir_num = 16
    data_dir = Path(f"~/data/goats/goats_{dir_num}").expanduser()
    data_file, beacon_loc_file = get_data_and_beacon_files(data_dir)

    dimension = 2
    filter_outlier_ranges = True
    parser = GoatsParser(data_file, beacon_loc_file, dimension, filter_outlier_ranges)  # type: ignore
    pyfg = parser.pyfg

    calibrations = get_range_measure_calibrations(pyfg)
    for cal in calibrations.values():
        print(cal)
