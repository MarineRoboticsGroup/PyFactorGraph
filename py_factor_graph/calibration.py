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
import matplotlib.pyplot as plt

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
    measured_dists: List[float] = field(default=[])
    true_dists: List[float] = field(default=[])

    def add_measurement(self, measured_dist: float, true_dist: float):
        """Add a measurement to the calibration.

        Args:
            measured_dist (float): The measured distance.
            true_dist (float): The true distance.
        """
        self.measured_dists.append(measured_dist)
        self.true_dists.append(true_dist)

    @property
    def measurements_vector(self):
        """Return the measurements as a vector."""
        return np.array(self.measured_dists)

    @property
    def true_distances_vector(self):
        """Return the true distances as a vector."""
        return np.array(self.true_dists)

    @property
    def num_measurements(self):
        """Return the number of measurements."""
        return len(self.measured_dists)

    @property
    def bias(self) -> float:
        """The bias of the range measurement."""
        return np.mean(self.measured_dists) - np.mean(self.true_dists)

    @property
    def covariance(self):
        """The covariance of the range measurement."""
        measures_np = np.array(self.measured_dists)
        true_dists_np = np.array(self.true_dists)
        return np.var(measures_np - true_dists_np)

    @property
    def stddev(self):
        return np.sqrt(self.covariance)

    def plot_error_vs_true_distance(self, ax=None, **kwargs):
        """Plot the error vs true distance.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Keyword arguments to pass to the plot
        """

        if ax is None:
            fig, ax = plt.subplots()

        true_dists = self.true_distances_vector
        measurements = self.measurements_vector
        ax.scatter(true_dists, measurements - true_dists, **kwargs)
        ax.set_xlabel("True distance")
        ax.set_ylabel("Error")
        ax.set_title(f"{self.var1} - {self.var2}")
        plt.show(block=True)
        return ax

    def __str__(self):
        """Return a string representation of the calibration."""
        return f"{self.var1} - {self.var2} ({self.num_measurements} measures): {self.bias} +- {self.stddev} stddev"


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
    pair_to_calibration_map: Dict[Tuple[str, str], RangePairCalibration] = {
        pair: RangePairCalibration(pair[0], pair[1])
        for pair in _get_all_variable_pairs()
    }

    # record the biases measured for each pair
    for range_measure in fg.range_measurements:
        var1, var2 = range_measure.association
        true_dist = _get_true_distance_between_variables(var1, var2)
        measured_dist = range_measure.dist

        pair1_var = var1[0] + "0"
        if var2.startswith("L"):
            pair2_var = var2
        else:
            pair2_var = var2[0] + "0"
        var_pair = (pair1_var, pair2_var)

        pair_to_calibration_map[var_pair].add_measurement(measured_dist, true_dist)

    return pair_to_calibration_map


if __name__ == "__main__":
    from py_factor_graph.parsing.parse_goats_data import (
        GoatsParser,
        get_data_and_beacon_files,
    )
    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file
    from pathlib import Path
    from py_factor_graph.modifiers import make_beacons_into_robot_trajectory

    # dir_num = 16
    # data_dir = Path(f"~/data/goats/goats_{dir_num}").expanduser()
    # data_file, beacon_loc_file = get_data_and_beacon_files(data_dir)
    # dimension = 2
    # filter_outlier_ranges = True
    # parser = GoatsParser(data_file, beacon_loc_file, dimension, filter_outlier_ranges)  # type: ignore
    # pyfg = parser.pyfg

    hat_experiment = "/home/alan/data/hat_data/16OCT2022/2022-10-16-12-14-46_terminate_added_cleaned_pyfg.pickle"
    pyfg = parse_pickle_file(hat_experiment)
    pyfg = make_beacons_into_robot_trajectory(pyfg)

    calibrations = get_range_measure_calibrations(pyfg)

    # get the only robot-robot calibration
    pair0 = ("A0", "B0")
    cal = calibrations[pair0]
    print(cal)

    # plot the error vs true distance
    cal.plot_error_vs_true_distance()
