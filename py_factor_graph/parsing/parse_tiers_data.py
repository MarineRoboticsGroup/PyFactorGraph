"""Parsing for the following data

This data is from the Turku University (Finland) TIERS lab. Specifically, Case 5, as detailed in
https://github.com/TIERS/uwb-relative-localization-dataset

"""
import numpy as np
from typing import Union, Tuple, Dict, List

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr, serialize_cdr
from rosbags.typesys.types import (
    sensor_msgs__msg__Range as Range,
    nav_msgs__msg__Odometry as Odometry,
    geometry_msgs__msg__PoseStamped as PoseStamped,
)

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import FGRangeMeasurement, PoseMeasurement2D
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    get_theta_from_rotation_matrix,
    get_measurement_precisions_from_covariances,
)
from py_factor_graph.parsing.range_measurement_calibration import (
    UncalibratedRangeMeasurement,
    get_inlier_set_of_range_measurements,
    get_linearly_calibrated_measurements,
)

from py_factor_graph.modifiers import skip_first_n_poses

import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)


NUM_TURTLES = 5

"""
Topic: /vrpn_client_node/tb{05}/pose
MsgType: geometry_msgs/msg/PoseStamped

Topic: /uwb/tof/n_{i+1}/n_{j+1}/distance
MsgType: sensor_msgs/msg/Range

Topic: /turtle{i+1:02d}/odom
MsgType: nav_msgs/msg/Odometry

"""

# topic names
DESIRED_UWB_TOPICS = [
    f"/uwb/tof/n_{i}/n_{j}/distance"
    for i in range(1, NUM_TURTLES + 1)
    for j in range(i + 1, NUM_TURTLES + 1)
]
DESIRED_ODOM_TOPICS = [f"/turtle{i:02d}/odom" for i in range(1, NUM_TURTLES + 1)]
DESIRED_GT_POSE_TOPICS = [
    f"/vrpn_client_node/tb{i:02d}/pose" for i in range(1, NUM_TURTLES + 1)
]

# handling the naming convention
TURTLE_IDX_TO_CHAR_MAP = {0: "A", 1: "B", 2: "C", 3: "L", 4: "D"}  # landmark
TURTLE_CHARS = list(TURTLE_IDX_TO_CHAR_MAP.values())
POSE_IDXS = [0, 1, 2, 4]  # landmark is turtle 3 so not a pose
CHAR_TO_TURTLE_IDX_MAP = {TURTLE_IDX_TO_CHAR_MAP[idx]: idx for idx in POSE_IDXS}
CHAR_TO_ROBOT_IDX_MAP = {
    TURTLE_IDX_TO_CHAR_MAP[idx]: POSE_IDXS.index(idx) for idx in POSE_IDXS
}
# 0 - bottom right
# 1 - bottom left
# 2 - top left
# 4 - top right

# the landmark is the 4th turtle
LANDMARK_IDX = 3


def _find_nearest_time_index(time_series: np.ndarray, target_time: float) -> int:
    """Find the index of the time series that is closest to the target time

    Args:
        time_series (List[float]): the time series
        target_time (float): the target time
        start_idx (int, optional): the index to start searching from. Defaults to 0.

    Returns:
        int: the index of the time series that is closest to the target time
    """
    time_diffs = np.abs(time_series - target_time)
    min_time_diff_idx = np.argmin(time_diffs)
    return int(min_time_diff_idx)


def _get_turtle_char_from_topic(topic_name: str) -> str:
    if topic_name in DESIRED_ODOM_TOPICS:
        topic_idx = DESIRED_ODOM_TOPICS.index(topic_name)
    elif topic_name in DESIRED_GT_POSE_TOPICS:
        topic_idx = DESIRED_GT_POSE_TOPICS.index(topic_name)
    else:
        raise ValueError(f"Cannot uniquely identify turtle idx from topic {topic_name}")
    return TURTLE_IDX_TO_CHAR_MAP[topic_idx]


def _rewrite_measurement_time(
    msg: Union[Odometry, Range, PoseStamped], new_time: float
) -> None:
    msg.header.stamp.sec = int(new_time)
    msg.header.stamp.nanosec = int((new_time - new_time) * 1e9)


def _get_measurement_time(msg: Union[Odometry, Range, PoseStamped]) -> float:
    time_sec = msg.header.stamp.sec
    time_nsec = msg.header.stamp.nanosec / 1e9
    time = time_sec + time_nsec
    return time


def _check_all_measurements_in_order(
    measurements: Union[List[PoseStamped], List[Odometry], List[Range]]
):
    prev_measurement_time = _get_measurement_time(measurements[0])
    for measurement in measurements[1:]:
        measurement_time = _get_measurement_time(measurement)
        assert (
            measurement_time > prev_measurement_time
        ), f"Measurement times not in order: {measurement_time} <= {prev_measurement_time}"
        prev_measurement_time = measurement_time


def _set_variables_and_odometry(
    pyfg: FactorGraphData,
    bag_reader: Reader,
    translational_stddev: float = 0.01,
    rotational_stddev: float = 0.005,
):
    trans_precision, rot_precision = get_measurement_precisions_from_covariances(
        trans_cov=translational_stddev**2, rot_cov=rotational_stddev**2
    )
    # only read the desired topics
    per_robot_odom_measures: Dict[str, List[Odometry]] = {
        robot_char: [] for robot_char in TURTLE_CHARS
    }
    per_robot_gt_poses: Dict[str, List[PoseStamped]] = {
        robot_char: [] for robot_char in TURTLE_CHARS
    }

    def _add_msg_to_collection(msg: Union[Odometry, PoseStamped], topic: str):
        turtle_char = _get_turtle_char_from_topic(topic)
        assert (
            turtle_char in TURTLE_CHARS
        ), f"Invalid turtle char {turtle_char} from topic {topic}"

        if topic in DESIRED_ODOM_TOPICS:
            assert isinstance(msg, Odometry), f"Msg {msg} is not of type Odometry"
            per_robot_odom_measures[turtle_char].append(msg)
        elif topic in DESIRED_GT_POSE_TOPICS:
            assert isinstance(msg, PoseStamped), f"Msg {msg} is not of type PoseStamped"
            per_robot_gt_poses[turtle_char].append(msg)
        else:
            raise ValueError(f"Invalid topic {topic}")

    pose_chain_topics = DESIRED_ODOM_TOPICS + DESIRED_GT_POSE_TOPICS
    pose_chain_connections = [
        connection
        for connection in bag_reader.connections
        if connection.topic in pose_chain_topics
    ]
    for connection, timestamp, raw_data in bag_reader.messages(
        connections=pose_chain_connections
    ):
        topic = connection.topic
        msg = deserialize_cdr(raw_data, connection.msgtype)
        _add_msg_to_collection(msg, topic)

    # verify that turtle "L" has no odometry measurements
    assert (
        len(per_robot_odom_measures["L"]) == 0
    ), f"Landmark turtle has {len(per_robot_odom_measures['L'])} odom measurements"

    # check that all measurements are in order
    for robot_char in TURTLE_CHARS:
        _check_all_measurements_in_order(per_robot_gt_poses[robot_char])

        # only check odometry measurements for robots, not the static turtlebot
        if robot_char != "L":
            _check_all_measurements_in_order(per_robot_odom_measures[robot_char])

    # add each robot's pose variables to the factor graph
    def _add_pose_variables(gt_poses: List[PoseStamped], robot_char: str):
        for pose_idx, pose in enumerate(gt_poses):
            pose_name = f"{robot_char}{pose_idx}"
            true_x = pose.pose.position.x
            true_y = pose.pose.position.y
            qx = pose.pose.orientation.x
            qy = pose.pose.orientation.y
            qz = pose.pose.orientation.z
            qw = pose.pose.orientation.w
            quat = np.array([qx, qy, qz, qw])
            rot3d = get_rotation_matrix_from_quat(quat)
            rot2d = rot3d[:2, :2]
            true_yaw = get_theta_from_rotation_matrix(rot2d)
            timestamp = _get_measurement_time(pose)
            pose_var = PoseVariable2D(pose_name, (true_x, true_y), true_yaw, timestamp)
            pyfg.add_pose_variable(pose_var)

    def _add_odom_measurements(
        rel_pose_measures: List[np.ndarray], odom_list: List[Odometry], robot_char: str
    ):
        robot_idx = CHAR_TO_ROBOT_IDX_MAP[robot_char]
        for odom_idx, rel_pose_measure in enumerate(rel_pose_measures):
            base_pose_name = f"{robot_char}{odom_idx}"
            to_pose_name = f"{robot_char}{odom_idx + 1}"
            timestamp = _get_measurement_time(odom_list[odom_idx])
            delta_x = rel_pose_measure[0, 2]
            delta_y = rel_pose_measure[1, 2]
            delta_yaw = get_theta_from_rotation_matrix(rel_pose_measure[:2, :2])
            pose_measurement = PoseMeasurement2D(
                base_pose=base_pose_name,
                to_pose=to_pose_name,
                x=delta_x,
                y=delta_y,
                theta=delta_yaw,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
                timestamp=timestamp,
            )
            pyfg.add_odom_measurement(robot_idx=robot_idx, odom_meas=pose_measurement)

    def _add_landmark_variable(pose_list: List[PoseStamped]):
        # average the positions of the landmark
        avg_x = np.mean([pose.pose.position.x for pose in pose_list])
        avg_y = np.mean([pose.pose.position.y for pose in pose_list])
        landmark_var = LandmarkVariable2D("L0", (float(avg_x), float(avg_y)))
        pyfg.add_landmark_variable(landmark_var)

    def _get_pose_chain_for_robot(
        odom_list: List[Odometry], gt_pose_list: List[PoseStamped]
    ) -> List[PoseStamped]:
        """Assumes that both lists are sorted by time of arrival and that there are
        many more ground truth poses than odometry measurements. Throws away any intermediate
        ground truth poses that do not have a corresponding odometry measurement.

        Args:
            odom_list (List[Odometry]): _description_
            gt_pose_list (List[PoseStamped]): _description_

        Returns:
            List[PoseStamped]: _description_
        """
        assert len(odom_list) > 0, "odom_list is empty"
        assert len(gt_pose_list) > len(
            odom_list
        ), "odom_list cannot be longer than gt_pose_list"
        # assert _get_measurement_time(odom_list[0]) > _get_measurement_time(
        #     gt_pose_list[0]
        # ), f"Odom list starts before gt pose list: {_get_measurement_time(odom_list[0])} <= {_get_measurement_time(gt_pose_list[0])}"
        logger.warning(
            "Not checking that odom measurements start after gt pose measurements"
        )

        gt_pose_chain = [gt_pose_list[0]]
        # iterate through the odometry measurements and find the nearest (time-wise) ground truth pose
        gt_pose_times = [_get_measurement_time(pose) for pose in gt_pose_list]
        gt_pose_time_arr = np.array(gt_pose_times)
        for odom in odom_list:
            odom_time = _get_measurement_time(odom)
            nearest_gt_pose_idx = _find_nearest_time_index(gt_pose_time_arr, odom_time)
            gt_pose_chain.append(gt_pose_list[nearest_gt_pose_idx])

        return gt_pose_chain

    def _get_relative_poses_from_odometry(
        odom_traj: List[Odometry],
    ) -> List[np.ndarray]:
        """Compute the relative poses from the composed odometry chain

        Args:
            odom_traj (List[Odometry]): the odometry chain

        Returns:
            List[PoseMeasurement2D]: a list of relative poses
        """

        def _get_2d_transformation_matrix(odom_msg: Odometry) -> np.ndarray:
            """Takes a 3D odometry message and returns a 2D transformation matrix

            Args:
                odom_msg (Odometry): _description_

            Returns:
                np.ndarray: _description_
            """
            x = odom_msg.pose.pose.position.x
            y = odom_msg.pose.pose.position.y
            quat_msg = odom_msg.pose.pose.orientation
            quat = np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w])
            rot = get_rotation_matrix_from_quat(quat)
            rot_2d = rot[:2, :2]

            T = np.eye(3)
            T[:2, :2] = rot_2d
            T[:2, 2] = np.array([x, y])
            return T

        def _get_relative_pose(
            prev_pose: np.ndarray, next_pose: np.ndarray
        ) -> np.ndarray:
            """Compute the relative pose between two poses

            Args:
                prev_pose (np.ndarray): the previous pose
                next_pose (np.ndarray): the next pose

            Returns:
                np.ndarray: the relative pose
            """
            return np.linalg.inv(prev_pose) @ next_pose

        # the first pose is the identity matrix
        rel_poses = []
        last_pose = np.eye(3)
        for odom in odom_traj:
            odom_pose = _get_2d_transformation_matrix(odom)
            rel_pose = _get_relative_pose(last_pose, odom_pose)
            rel_poses.append(rel_pose)
            last_pose = odom_pose

        return rel_poses

    for robot_char in TURTLE_CHARS:
        odom_list = per_robot_odom_measures[robot_char]
        gt_pose_list = per_robot_gt_poses[robot_char]
        if robot_char == "L":
            _add_landmark_variable(gt_pose_list)
        else:
            pose_chain = _get_pose_chain_for_robot(odom_list, gt_pose_list)
            rel_pose_measure_list = _get_relative_poses_from_odometry(odom_list)
            assert len(pose_chain) == len(rel_pose_measure_list) + 1
            _add_pose_variables(pose_chain, robot_char)
            _add_odom_measurements(rel_pose_measure_list, odom_list, robot_char)


def _add_range_measurements(pyfg: FactorGraphData, bag_reader: Reader):
    logger.warning("Adding range measurements must be done after adding pose variables")

    def _get_robot_idx_pair_from_topic(topic: str) -> Tuple[int, int]:
        assert topic in DESIRED_UWB_TOPICS, f"Invalid topic {topic}"
        # Topic: /uwb/tof/n_{i+1}/n_{j+1}/distance

        topic_parts = topic.split("/")
        # topic_parts = ['', 'uwb', 'tof', 'n_{i+1}', 'n_{j+1}', 'distance']

        i_part = topic_parts[-3]
        j_part = topic_parts[-2]
        assert i_part.startswith("n_") and j_part.startswith(
            "n_"
        ), f"Invalid topic {topic}"
        # i_part = 'n_{i+1}' j_part = 'n_{j+1}'

        i_idx = int(i_part[2:]) - 1
        j_idx = int(j_part[2:]) - 1
        # i_idx = i, j_idx = j

        return (i_idx, j_idx)

    # compile all of the ranges
    uwb_connections = [
        connection
        for connection in bag_reader.connections
        if connection.topic in DESIRED_UWB_TOPICS
    ]
    uwb_pair_to_measurements: Dict[Tuple[int, int], List[Range]] = {
        (i, j): [] for i in range(NUM_TURTLES) for j in range(i + 1, NUM_TURTLES)
    }
    for connection, timestamp, raw_data in bag_reader.messages(
        connections=uwb_connections
    ):
        msg = deserialize_cdr(raw_data, connection.msgtype)
        assert isinstance(msg, Range), f"Msg {msg} is not of type Range"
        topic = connection.topic
        i_idx, j_idx = _get_robot_idx_pair_from_topic(topic)

        # because all measurements are redundantly copied (i,j) and (j, i), we only consider ones that are
        # from the robot with the smaller index
        if j_idx < i_idx:
            continue

        uwb_pair_to_measurements[(i_idx, j_idx)].append(msg)

    # make sure that all measurements are in order
    for _, measurements in uwb_pair_to_measurements.items():
        _check_all_measurements_in_order(measurements)

    # gather the timestamps for all pose variables to properly associate the range measurements
    def _get_list_of_pose_timestamps_per_robot(robot_idx: int) -> List[float]:
        timestamps = [
            pyfg.pose_variables[robot_idx][i].timestamp
            for i in range(len(pyfg.pose_variables[robot_idx]))
        ]
        return timestamps  # type: ignore

    pose_var_timestamps: Dict[int, np.ndarray] = {
        POSE_IDXS[i]: np.asarray(_get_list_of_pose_timestamps_per_robot(i))
        for i in range(len(POSE_IDXS))
    }

    # iterate through the measurements and collect them as uncalibrated range measurements
    def _find_corresponding_variable_symbol_and_index(
        robot_idx: int, timestamp: float
    ) -> str:
        if robot_idx == LANDMARK_IDX:
            return pyfg.landmark_variables[0].name

        pose_timestamps = pose_var_timestamps[robot_idx]
        pose_idx = _find_nearest_time_index(pose_timestamps, timestamp)

        if robot_idx == pyfg.num_robots:
            robot_search_idx = robot_idx - 1
        else:
            robot_search_idx = robot_idx
        pose_var_name = pyfg.pose_variables[robot_search_idx][pose_idx].name
        return pose_var_name

    # get the true positions of all variables
    true_position_map = pyfg.variable_true_positions_dict

    def _get_true_dist_between_variables(var1_name: str, var2_name: str) -> float:
        v1_pos = np.array(true_position_map[var1_name])
        v2_pos = np.array(true_position_map[var2_name])
        return float(np.linalg.norm(v1_pos - v2_pos))

    uncalibrated_measurements: Dict[
        Tuple[int, int], List[UncalibratedRangeMeasurement]
    ] = {key: [] for key in uwb_pair_to_measurements.keys()}
    for (i_idx, j_idx), measurements in uwb_pair_to_measurements.items():
        logger.debug(
            f"UWB pair: ({i_idx}, {j_idx}) has {len(measurements)} measurements"
        )
        for measurement in measurements:
            measurement_time = _get_measurement_time(measurement)

            if i_idx == LANDMARK_IDX:
                i_sym = pyfg.landmark_variables[0].name
            else:
                i_sym = _find_corresponding_variable_symbol_and_index(
                    i_idx, measurement_time
                )

            if j_idx == LANDMARK_IDX:
                j_sym = pyfg.landmark_variables[0].name
            else:
                j_sym = _find_corresponding_variable_symbol_and_index(
                    j_idx, measurement_time
                )

            # if i_sym is "L0" we need to swap the variables
            if i_sym == "L0":
                i_sym, j_sym = j_sym, i_sym

            association = (i_sym, j_sym)
            uncalibrated_measurement = UncalibratedRangeMeasurement(
                association=association,
                dist=measurement.range,
                timestamp=measurement_time,
                true_dist=_get_true_dist_between_variables(i_sym, j_sym),
            )
            uncalibrated_measurements[(i_idx, j_idx)].append(uncalibrated_measurement)

    # calibrate the range measurements and gather them into a single list
    for (i_idx, j_idx), uncal_measurements in uncalibrated_measurements.items():
        logger.debug(
            f"UWB pair: ({i_idx}, {j_idx}) has {len(uncal_measurements)} measurements"
        )
        inlier_measurements = get_inlier_set_of_range_measurements(
            uncal_measurements, inlier_stddev_threshold=2, show_outlier_rejection=False
        )
        calibrated_measurements = get_linearly_calibrated_measurements(
            inlier_measurements
        )
        logger.debug(
            f"After calibration, there are {len(calibrated_measurements)} measurements"
        )
        calibrated_measurements_by_association: Dict[
            Tuple[str, str], List[FGRangeMeasurement]
        ] = {meas.association: [] for meas in calibrated_measurements}
        for calibrated_measurement in calibrated_measurements:
            calibrated_measurements_by_association[
                calibrated_measurement.association
            ].append(calibrated_measurement)

        for association, cal_measures in calibrated_measurements_by_association.items():
            if len(cal_measures) == 1:
                # there is only one measurement
                calibrated_measurement = cal_measures[0]
                pyfg.add_range_measurement(calibrated_measurement)
            else:
                # extract an averaged measurement from the set of measurements
                avg_dist = float(np.mean([meas.dist for meas in cal_measures]))
                timestamps = [meas.timestamp for meas in cal_measures if meas.timestamp]
                avg_timestamp = float(np.mean(timestamps))
                stddev = cal_measures[0].stddev
                avg_measurement = FGRangeMeasurement(
                    association=association,
                    dist=avg_dist,
                    stddev=stddev,
                    timestamp=avg_timestamp,
                )
                pyfg.add_range_measurement(avg_measurement)


def parse_tiers_uwb_bag(bag_dir: str) -> FactorGraphData:
    """Parse the bagfile from TIERS UWB dataset. Note that
    turtle4 is static, and we will thus treat it as a landmark.

    Args:
        bag_dir (str): the directory containing the ROS2 bag data

    Returns:
        FactorGraphData: factor graph data
    """

    fg = FactorGraphData(dimension=2)
    with Reader(bag_dir) as reader:
        _set_variables_and_odometry(fg, reader)
        _add_range_measurements(fg, reader)

    # we are going to skip the first "n" poses because the odometry is bad at the beginning
    num_poses_to_skip = 100
    logger.warning(
        f"Skipping the first {num_poses_to_skip} poses because the odometry is bad at the beginning"
    )
    fg = skip_first_n_poses(fg, num_poses_to_skip)
    return fg


if __name__ == "__main__":
    from os.path import expanduser

    bag_path = expanduser("~/experimental_data/tiers/ros2/")
    fg = parse_tiers_uwb_bag(bag_path)
    fg.print_summary()
