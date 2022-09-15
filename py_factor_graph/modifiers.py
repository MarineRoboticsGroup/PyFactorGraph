"""These are functions that operate on the factor graphs to create new
factor graphs from the original.

Examples:
    1) a modifier that simulates range measurements between different poses
    2) a modifier that splits a single-robot factor graph into a multi-robot one
"""

from typing import Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
import copy
import itertools
from attrs import define, field

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
from py_factor_graph.utils.matrix_utils import (
    _check_transformation_matrix,
    get_random_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_translation_from_transformation_matrix,
)
from py_factor_graph.utils.attrib_utils import (
    probability_validator,
    positive_float_validator,
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
class RangeMeasurementModel:
    """This class defines a sensor model for range measurements

    Args:
    """

    sensing_horizon: float = field(validator=positive_float_validator)
    range_stddev: float = field(validator=positive_float_validator)
    measurement_prob: float = field(validator=probability_validator)

    def make_measurement(
        self, association: Tuple[str, str], dist: float, timestamp: Optional[float]
    ) -> FGRangeMeasurement:
        """Makes a range measurement

        Args:
            dist (float): the distance between the two poses

        Returns:
            FGRangeMeasurement: the measurement
        """
        assert (
            0 < dist <= self.sensing_horizon
        ), f"The distance {dist} is not in the sensing horizon"
        noisy_dist = np.random.normal(dist, self.range_stddev)
        if noisy_dist < 0:
            logger.warning(f"Negative noisy distance {noisy_dist} for dist {dist}")
            noisy_dist = 0.0

        return FGRangeMeasurement(
            association=association,
            dist=noisy_dist,
            stddev=self.range_stddev,
            timestamp=timestamp,
        )


def _dist_between_variables(
    var1: Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES],
    var2: Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES],
) -> float:
    """Returns the distance between two variables"""
    if isinstance(var1, PoseVariable2D) or isinstance(var1, PoseVariable3D):
        pos1 = var1.position_vector
    elif isinstance(var1, LandmarkVariable2D) or isinstance(var1, LandmarkVariable3D):
        pos1 = np.array(var1.true_position)
    else:
        raise ValueError(f"Variable {var1} not supported")

    if isinstance(var2, PoseVariable2D) or isinstance(var2, PoseVariable3D):
        pos2 = var2.position_vector
    elif isinstance(var2, LandmarkVariable2D) or isinstance(var2, LandmarkVariable3D):
        pos2 = np.array(var2.true_position)
    else:
        raise ValueError(f"Variable {var2} not supported")

    dist = np.linalg.norm(pos1 - pos2)
    return dist


def add_landmark_at_position(
    fg: FactorGraphData,
    landmark_position: np.ndarray,
    range_measurement_model: RangeMeasurementModel,
) -> FactorGraphData:
    """Adds landmark at a random positions in the factor graph and simulates
    range measurements between the landmarks and the robot poses.

    Args:
        fg (FactorGraphData): the factor graph to modify
        num_landmarks (int): the number of landmarks to add

    Returns:
        FactorGraphData: a new factor graph with the added landmarks
    """
    new_fg = copy.deepcopy(fg)
    dim = fg.dimension
    assert landmark_position.shape == (dim, 1) or landmark_position.shape == (dim,)

    new_landmark_name = f"L{new_fg.num_landmarks}"
    new_landmark: LANDMARK_VARIABLE_TYPES
    if dim == 2:
        new_landmark = LandmarkVariable2D(new_landmark_name, landmark_position)
    elif dim == 3:
        new_landmark = LandmarkVariable3D(new_landmark_name, landmark_position)
    else:
        raise ValueError(f"Dimension {dim} not supported")
    new_fg.add_landmark_variable(new_landmark)

    for pose_chain in new_fg.pose_variables:
        for pose in pose_chain:
            pose_to_landmark_dist = _dist_between_variables(pose, new_landmark)
            if (
                pose_to_landmark_dist <= range_measurement_model.sensing_horizon
                and np.random.rand() < range_measurement_model.measurement_prob
            ):
                association = (pose.name, new_landmark_name)
                range_measure = range_measurement_model.make_measurement(
                    association, pose_to_landmark_dist, pose.timestamp
                )
                new_fg.add_range_measurement(range_measure)

    return new_fg


def add_landmark_at_trajectory_center(
    fg: FactorGraphData,
    range_measurement_model: RangeMeasurementModel,
) -> FactorGraphData:
    """Adds landmark at the center of the trajectory and simulates
    range measurements between the landmarks and the robot poses.

    Args:
        fg (FactorGraphData): the factor graph to modify
        num_landmarks (int): the number of landmarks to add

    Returns:
        FactorGraphData: a new factor graph with the added landmarks
    """
    dim = fg.dimension
    assert (
        fg.num_robots == 1
    ), f"Only single robot factor graphs supported, got {fg.num_robots} robots"
    trajectory_center = np.array([0.0] * dim)

    for pose_chain in fg.pose_variables:
        for pose in pose_chain:
            trajectory_center += pose.position_vector
    trajectory_center /= fg.num_poses
    logger.info(f"Added landmark at trajectory center {trajectory_center}")
    return add_landmark_at_position(fg, trajectory_center, range_measurement_model)


def add_random_landmarks(
    fg: FactorGraphData, num_landmarks: int, range_model: RangeMeasurementModel
) -> FactorGraphData:
    """Returns the

    Args:
        fg (FactorGraphData): _description_
        num_landmarks (int): _description_

    Returns:
        FactorGraphData: _description_
    """
    new_fg = copy.deepcopy(fg)
    for _ in range(num_landmarks):
        rand_x = np.random.uniform(fg.x_min, fg.x_max)
        rand_y = np.random.uniform(fg.y_min, fg.y_max)
        rand_z = np.random.uniform(fg.z_min, fg.z_max)

        if fg.dimension == 2:
            landmark_position = np.array([rand_x, rand_y])
        elif fg.dimension == 3:
            landmark_position = np.array([rand_x, rand_y, rand_z])
        else:
            raise ValueError(f"Dimension {fg.dimension} not supported")

        new_fg = add_landmark_at_position(new_fg, landmark_position, range_model)
    return new_fg


def remove_loop_closures(
    fg: FactorGraphData,
) -> FactorGraphData:
    """Removes loop closures from the factor graph

    Args:
        fg (FactorGraphData): the factor graph to modify

    Returns:
        FactorGraphData: a new factor graph with the loop closures removed
    """
    new_fg = copy.deepcopy(fg)
    new_fg.loop_closure_measurements.clear()
    return new_fg


def reduce_number_of_loop_closures(
    fg: FactorGraphData, percent_to_keep: float
) -> FactorGraphData:
    """Only keeps a certain percentage of the loop closures in the factor graph

    Args:
        fg (FactorGraphData): the factor graph to modify
        percent_to_keep (float): the percentage of loop closures to keep

    Returns:
        FactorGraphData: a new factor graph with the loop closures removed
    """
    new_fg = copy.deepcopy(fg)
    new_fg.loop_closure_measurements = random.sample(
        new_fg.loop_closure_measurements,
        int(len(new_fg.loop_closure_measurements) * percent_to_keep),
    )
    return new_fg


def split_single_robot_into_multi(
    fg: FactorGraphData, num_robots: int
) -> FactorGraphData:
    """Takes a single-robot factor graph and splits it into a multi-robot one.

    Args:
        fg: The single-robot factor graph.
        num_robots: The number of robots to split the factor graph into.

    Returns:
        The multi-robot factor graph.
    """
    assert (
        fg.num_robots == 1
    ), f"Expected a single-robot factor graph, but got {fg.num_robots}."

    num_total_poses = fg.num_poses
    multi_fg = FactorGraphData(fg.dimension)
    old_to_new_pose_name_mapping: Dict[str, str] = {}

    def _get_pose_chain_bound_idxs() -> List[Tuple[int, int]]:
        # set the start/stop indices for each robot using numpy
        pose_chain_indices = np.linspace(
            start=0, stop=num_total_poses, num=num_robots + 1, dtype=int
        )
        robot_pose_chain_bounds = list(
            zip(pose_chain_indices[:-1], pose_chain_indices[1:])
        )
        return robot_pose_chain_bounds

    def _copy_pose_variable_with_new_name(
        old_pose: POSE_VARIABLE_TYPES, new_name: str
    ) -> POSE_VARIABLE_TYPES:
        if isinstance(old_pose, PoseVariable2D):
            return PoseVariable2D(
                new_name,
                old_pose.true_position,
                old_pose.true_theta,
                old_pose.timestamp,
            )
        elif isinstance(old_pose, PoseVariable3D):
            return PoseVariable3D(
                new_name,
                old_pose.true_position,
                old_pose.true_rotation,
                old_pose.timestamp,
            )
        else:
            raise ValueError(f"Unknown pose type: {type(old_pose)}")

    def _add_pose_variables() -> None:
        # add the pose variables
        pose_chain = fg.pose_variables[0]
        pose_chain_bound_idxs = _get_pose_chain_bound_idxs()

        for robot_idx, (pose_chain_start, pose_chain_end) in enumerate(
            pose_chain_bound_idxs
        ):
            robot_pose_chain = pose_chain[pose_chain_start:pose_chain_end]
            robot_char = get_robot_char_from_number(robot_idx)

            for pose_idx, old_pose in enumerate(robot_pose_chain):
                pose_name = f"{robot_char}{pose_idx}"
                new_pose = _copy_pose_variable_with_new_name(old_pose, pose_name)
                old_to_new_pose_name_mapping[old_pose.name] = pose_name
                multi_fg.add_pose_variable(new_pose)

    def _copy_odom_measurement_with_new_frames(
        old_measure: POSE_MEASUREMENT_TYPES, from_frame: str, to_frame: str
    ) -> POSE_MEASUREMENT_TYPES:
        if isinstance(old_measure, PoseMeasurement2D):
            return PoseMeasurement2D(
                from_frame,
                to_frame,
                old_measure.x,
                old_measure.y,
                old_measure.theta,
                old_measure.translation_precision,
                old_measure.rotation_precision,
                old_measure.timestamp,
            )
        elif isinstance(old_measure, PoseMeasurement3D):
            return PoseMeasurement3D(
                from_frame,
                to_frame,
                old_measure.translation,
                old_measure.rotation,
                old_measure.translation_precision,
                old_measure.rotation_precision,
                old_measure.timestamp,
            )
        else:
            raise ValueError(f"Unknown pose type: {type(old_measure)}")

    def _add_odom_measurements() -> None:
        # add the odometry measurements
        pose_chain_bound_idxs = _get_pose_chain_bound_idxs()
        odom_chain = fg.odom_measurements[0]
        for robot_idx, (pose_chain_start, pose_chain_end) in enumerate(
            pose_chain_bound_idxs
        ):
            robot_odom_chain = odom_chain[pose_chain_start : pose_chain_end - 1]
            robot_char = get_robot_char_from_number(robot_idx)

            for odom_idx, odom in enumerate(robot_odom_chain):
                from_pose = f"{robot_char}{odom_idx}"
                to_pose = f"{robot_char}{odom_idx+1}"
                new_measure = _copy_odom_measurement_with_new_frames(
                    odom, from_pose, to_pose
                )
                multi_fg.add_odom_measurement(robot_idx, new_measure)

    def _add_loop_closures() -> None:
        # add the loop closures
        for loop_closure in fg.loop_closure_measurements:
            old_from_frame = loop_closure.base_pose
            old_to_frame = loop_closure.to_pose
            new_loop_closure = _copy_odom_measurement_with_new_frames(
                loop_closure,
                old_to_new_pose_name_mapping[old_from_frame],
                old_to_new_pose_name_mapping[old_to_frame],
            )
            multi_fg.add_loop_closure(new_loop_closure)

    _add_pose_variables()
    _add_odom_measurements()
    _add_loop_closures()

    assert multi_fg.num_robots == num_robots
    return multi_fg


def add_inter_robot_range_measurements(
    fg: FactorGraphData,
    range_model: RangeMeasurementModel,
) -> FactorGraphData:
    """Adds range measurements between robots within a given sensing horizon

    Args:
        fg (FactorGraphData): the original factor graph to modify
        sensing_horizon (float): the sensing horizon for the range measurements
        measurement_prob (float, optional): the probability of a measurement
            being added. Defaults to 0.3.
        range_stddev (float, optional): the standard deviation of the range

    Returns:
        FactorGraphData: a new factor graph with the added measurements
    """
    assert (
        fg.num_robots > 1
    ), "Cannot add inter-robot measurements to a single robot factor graph."
    sensing_horizon = range_model.sensing_horizon
    measurement_prob = range_model.measurement_prob
    range_stddev = range_model.range_stddev
    logger.debug(
        f"Adding inter-robot ranges currently assumes that timesteps are matched across pose chains"
    )

    new_fg = copy.deepcopy(fg)

    def _poses_have_same_timestamp(
        pose1: POSE_VARIABLE_TYPES, pose2: POSE_VARIABLE_TYPES
    ) -> bool:
        return pose1.timestamp == pose2.timestamp

    for pose_chain1, pose_chain2 in itertools.combinations(new_fg.pose_variables, 2):
        chain_1_idx, chain_2_idx = 0, 0
        chain_1_end, chain_2_end = len(pose_chain1), len(pose_chain2)

        while chain_1_idx < chain_1_end or chain_2_idx < chain_2_end:
            # use these selectors to avoid out of bounds errors
            pose1_selector = min(chain_1_idx, chain_1_end - 1)
            pose2_selector = min(chain_2_idx, chain_2_end - 1)
            pose1 = pose_chain1[pose1_selector]
            pose2 = pose_chain2[pose2_selector]

            same_timestamp = _poses_have_same_timestamp(pose1, pose2)
            one_pose_at_end = chain_1_idx == chain_1_end or chain_2_idx == chain_2_end
            if not same_timestamp and not one_pose_at_end:
                err = (
                    f"The timestamps are mismatched: {pose1.timestamp} != {pose2.timestamp}"
                    " and neither pose is at the end of its chain."
                )
                raise ValueError(err)

            dist = _dist_between_variables(pose1, pose2)

            if dist <= sensing_horizon and np.random.rand() < measurement_prob:
                association = (pose1.name, pose2.name)
                range_measure = range_model.make_measurement(
                    association, dist, range_stddev
                )
                new_fg.add_range_measurement(range_measure)

            # increment the indices
            if chain_1_idx < chain_1_end:
                chain_1_idx += 1
            if chain_2_idx < chain_2_end:
                chain_2_idx += 1

    num_range_measurements = new_fg.num_range_measurements
    assert (
        num_range_measurements > 0
    ), "No range measurements were added to the factor graph."
    return new_fg


def make_single_robot_into_multi_via_transform(
    fg: FactorGraphData, num_robots: int
) -> FactorGraphData:
    """Generates many similar trajectories (copies perturbed by a transform) from
    a single robot trajectory

    Args:
        fg (FactorGraphData): the factor graph to modify
        transform (np.ndarray): the offset to apply to the single robot trajectory

    Returns:
        FactorGraphData: a new factor graph with the modified trajectory
    """
    assert (
        fg.num_robots == 1
    ), "Cannot make a multi-robot factor graph into a single robot factor graph."
    dim = fg.dimension
    new_fg = copy.deepcopy(fg)

    def _get_pose_transform(idx: int) -> np.ndarray:
        return get_random_transformation_matrix(dim)

    def _make_new_pose(
        pose: POSE_VARIABLE_TYPES, transform: np.ndarray, new_name: str
    ) -> POSE_VARIABLE_TYPES:
        new_transform = pose.transformation_matrix @ transform
        new_translation = get_translation_from_transformation_matrix(new_transform)
        if isinstance(pose, PoseVariable2D):
            new_theta = get_theta_from_transformation_matrix(new_transform)
            return PoseVariable2D(new_name, new_translation, new_theta, pose.timestamp)
        elif isinstance(pose, PoseVariable3D):
            new_rot = get_rotation_matrix_from_transformation_matrix(new_transform)
            return PoseVariable3D(new_name, new_translation, new_rot, pose.timestamp)
        else:
            raise ValueError(f"Invalid pose type: {type(pose)}")

    # make transformed poses and copy over odometry measurements
    original_pose_chain = new_fg.pose_variables[0]
    original_odom_chain = new_fg.odom_measurements[0]
    for robot_idx in range(num_robots):
        robot_char = get_robot_char_from_number(robot_idx)

        # poses
        transform = _get_pose_transform(robot_idx)
        for pose_idx, pose in enumerate(original_pose_chain):
            new_pose_name = f"{robot_char}{pose_idx}"
            new_pose = _make_new_pose(pose, transform, new_pose_name)
            new_fg.add_pose_variable(new_pose)

        # odometry
        for odom_idx, odom in enumerate(original_odom_chain):
            new_base_frame = f"{robot_char}{odom_idx}"
            new_to_frame = f"{robot_char}{odom_idx + 1}"
            if isinstance(odom, PoseMeasurement2D):
                new_odom_2d = PoseMeasurement2D(
                    new_base_frame,
                    new_to_frame,
                    odom.x,
                    odom.y,
                    odom.theta,
                    odom.translation_precision,
                    odom.rotation_precision,
                    odom.timestamp,
                )
                new_fg.add_odom_measurement(robot_idx, new_odom_2d)
            elif isinstance(odom, PoseMeasurement3D):
                new_odom_3d = PoseMeasurement3D(
                    new_base_frame,
                    new_to_frame,
                    odom.translation,
                    odom.rotation,
                    odom.translation_precision,
                    odom.rotation_precision,
                    odom.timestamp,
                )
                new_fg.add_odom_measurement(robot_idx, new_odom_3d)
            else:
                raise ValueError(f"Invalid odom type: {type(odom)}")

    return new_fg


def split_single_robot_into_multi_and_add_ranges(
    fg: FactorGraphData,
    num_robots: int,
    range_model: RangeMeasurementModel,
) -> FactorGraphData:
    """
    Splits a single robot factor graph into multiple robots and adds range
    measurements between the robots.

    Args:
        fg (FactorGraphData): the factor graph to modify
        num_robots (int): the number of robots to split the trajectory into
        range_model (RangeMeasurementModel): the range measurement model to use

    Returns:
        FactorGraphData: a new factor graph with the modified trajectory
    """
    new_fg = split_single_robot_into_multi(fg, num_robots)
    new_fg = add_inter_robot_range_measurements(new_fg, range_model)
    return new_fg


def set_all_precisions(
    fg: FactorGraphData,
    trans_precision: float,
    rot_precision: float,
    range_precision: float,
) -> FactorGraphData:
    """Sets all precisions across the factor graph to a given value per type of measurement.

    Args:
        fg (FactorGraphData): the factor graph to modify
        trans_precision (float): the translation precision to set


    Returns:
        FactorGraphData: a new factor graph with the modified precisions
    """
    new_fg = copy.deepcopy(fg)
    range_stddev = (1.0 / range_precision) ** 0.5
    for odom in new_fg.odom_measurements:
        for odom_meas in odom:
            odom_meas.translation_precision = trans_precision
            odom_meas.rotation_precision = rot_precision
    for range_meas in new_fg.range_measurements:
        range_meas.stddev = range_stddev
    return new_fg


if __name__ == "__main__":
    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file
    import os

    np.random.seed(0)

    def _get_pickle_file_path_in_dir(dir_path: str) -> str:
        files = os.listdir(dir_path)
        pickle_files = [f for f in files if f.endswith(".pickle")]
        assert (
            len(pickle_files) == 1
        ), f"Expected 1 pickle file, but got {len(pickle_files)}"
        pickle_file = pickle_files[0]
        pickle_file_path = os.path.join(dir_path, pickle_file)
        return pickle_file_path

    options = [
        # "cubicle",
        "garage",
        "grid",
        "smallGrid",
        "sphere2500",
        # "sphereBigNoise",
        "tinyGrid",
        "torus",
    ]
    sensing_horizon = {
        "cubicle": 20.0,
        "garage": 250.0,
        "grid": 30.0,
        "smallGrid": 15.0,
        "sphere2500": 100.0,
        "sphereBigNoise": 350.0,
        "tinyGrid": 15.0,
        "torus": 12.0,
    }

    modifications = {
        1: "split_single_to_multi",
        2: "split_single_to_multi_with_ranges",
        3: "add_landmark_at_center",
        4: "add_landmark_at_center_and_remove_loop_closures",
        5: "add_random_landmarks",
    }
    mod_choice = 4
    mod = modifications[mod_choice]

    num_random_landmarks = 5
    range_stddev = 0.75
    range_prob = 0.2
    num_robots = 3

    for option in options:
        # if "sphere" not in option:
        #     continue

        fg_data_dir = f"/home/alan/data/slam-data-sets/g2o/se_sync_gt/{option}"
        fg_data_file = _get_pickle_file_path_in_dir(fg_data_dir)
        new_fg_path = fg_data_file.replace("se_sync_gt", f"se_sync_gt_{mod}")

        if os.path.isfile(new_fg_path):
            user_skip = "n"
            while user_skip not in ["y", "n"]:
                user_skip = input(f"File already exists: {new_fg_path}. Skip? [y/n] ")
            if user_skip == "y":
                continue

        range_measurement_model = RangeMeasurementModel(  # type: ignore
            sensing_horizon[option], range_stddev, range_prob
        )
        fg = parse_pickle_file(fg_data_file)

        if mod == "add_landmark_at_center":
            new_fg = add_landmark_at_trajectory_center(fg, range_measurement_model)
        elif mod == "split_single_to_multi":
            new_fg = split_single_robot_into_multi(fg, num_robots=num_robots)
        elif mod == "split_single_to_multi_with_ranges":
            new_fg = split_single_robot_into_multi_and_add_ranges(
                fg, num_robots=num_robots, range_model=range_measurement_model
            )
        elif mod == "add_landmark_at_center_and_remove_loop_closures":
            int_fg = add_landmark_at_trajectory_center(fg, range_measurement_model)
            # new_fg = reduce_number_of_loop_closures(int_fg, percent_to_keep=0.1)
            new_fg = remove_loop_closures(int_fg)
        elif mod == "add_random_landmarks":
            new_fg = add_random_landmarks(
                fg,
                num_landmarks=num_random_landmarks,
                range_model=range_measurement_model,
            )
            new_fg = remove_loop_closures(new_fg)
        else:
            raise ValueError(f"Invalid mod: {mod_choice}")

        # new_fg.plot_odom_precisions()
        odom_precisions = new_fg.odom_precisions
        min_trans_precision = min(odom_precisions, key=lambda t: t[0])[0]
        min_rot_precision = min(odom_precisions, key=lambda t: t[1])[1]
        max_trans_precision = max(odom_precisions, key=lambda t: t[0])[0]
        max_rot_precision = max(odom_precisions, key=lambda t: t[1])[1]
        logger.info(
            f"Min translation precision: {min_trans_precision} Min rot precision: {min_rot_precision}"
        )
        logger.info(
            f"Max translation precision: {max_trans_precision} Max rot precision: {max_rot_precision}"
        )
        new_fg.print_summary()
        new_fg._save_to_pickle_format(new_fg_path)
