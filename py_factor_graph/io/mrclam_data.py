"""
This parser is designed to parse the UTIAS Multi-Robot Cooperative Localization and Mapping Dataset (MRCLAM)
The data is available at: http://asrl.utias.utoronto.ca/datasets/mrclam/index.html

There are 5 robots in each dataset and we assume all files exist for each robot, as extracted from the zip file.
"""

import argparse
import logging
import os
from typing import Tuple

import coloredlogs
import numpy as np
import pandas as pd
from tqdm import tqdm

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.measurements import (
    FGRangeMeasurement,
    PoseMeasurement2D,
    PoseToLandmarkMeasurement2D,
)
from py_factor_graph.priors import LandmarkPrior2D
from py_factor_graph.utils.name_utils import get_robot_char_from_number
from py_factor_graph.variables import LandmarkVariable2D, PoseVariable2D

np.set_printoptions(formatter={"all": lambda x: str(x)})

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


class Trajectory:
    """
    This class handles the storage and interpolation of the robot's trajectory to create
    ground truth poses from un-synchronized data.
    """

    def __init__(self, data) -> None:
        """
        Gets a np array with timestamp, x, y, theta and creates a trajectory object
        """
        self.data = data
        self.min_ts = np.min(self.data[:, 0])
        self.max_ts = np.max(self.data[:, 0])
        assert self.data.shape[1] == 4, f"Expected 4 columns in {data}"

        # Sort by timestamp
        self.data = self.data[self.data[:, 0].argsort()]

    def at_timestamp(self, timestamp: float) -> Tuple[np.ndarray, float]:
        """
        Returns the pose at the given timestamp by interpolating between the closest two poses
          and the change in time from the closest timestamp
        """
        return (
            np.interp(timestamp, self.data[:, 0], self.data[:, 1:], axis=0),
            np.abs(self.data[:, 0] - timestamp).min(),
        )


def get_robot_name(robot_num: int):
    """
    Convert robot number 1-5 to A-E
    """
    assert robot_num >= 1 and robot_num <= 5, f"Invalid robot number {robot_num}"
    return get_robot_char_from_number(robot_num - 1)


def load_robot_gt(fname):
    """
    Loads the ground truth trajectory of a robot from the given file and returns a Trajectory object
    """
    df = parse_whitespace_file(fname)
    assert df.shape[1] == 4, f"Expected 4 columns in {fname}"
    df.columns = ["timestamp", "x", "y", "theta"]
    return Trajectory(df.to_numpy())


def add_landmarks(
    fg: FactorGraphData, dirpath: str, landmark_stddev: float, add_landmark_prior: bool
):
    # Add all landmarks as vars, and add a prior if requested
    landmark_gt_filepath = os.path.join(dirpath, "Landmark_Groundtruth.dat")
    landmark_gt_df = parse_whitespace_file(landmark_gt_filepath)
    assert landmark_gt_df.shape[1] == 5, f"Expected 5 columns in {landmark_gt_filepath}"
    landmark_gt_df.columns = ["landmark_id", "x", "y", "x_stddev", "y_stddev"]
    logger.info("Loaded %d landmarks", landmark_gt_df.shape[0])
    if add_landmark_prior:
        logger.info("Adding landmark priors with stddev %f", landmark_stddev)

    for _, row in landmark_gt_df.iterrows():
        landmark_name = f"L{int(row['landmark_id'])}"
        landmark_var = LandmarkVariable2D(landmark_name, (row["x"], row["y"]))
        fg.add_landmark_variable(landmark_var)

        if add_landmark_prior:
            landmark_prior = LandmarkPrior2D(
                landmark_name,
                (row["x"], row["y"]),
                landmark_stddev,
            )
            fg.add_landmark_prior(landmark_prior)


def get_all_measurements(
    barcode_fname: str,
    data_dir: str,
):
    """
    Returns a dataframe with all range-bearing measurements from all robots in ascending timestamp

    Filters out measurements with unknown barcodes
    """
    # Load the barcode data
    barcode_df = parse_whitespace_file(barcode_fname)
    barcode_df.columns = ["subject", "barcode"]
    barcode_to_var_name = {}
    # Convert it into a dict between barcode and robot or landmark name
    for _, row in barcode_df.iterrows():
        if row["subject"] <= 5:  # Robots are numbered 1-5
            barcode_to_var_name[row["barcode"]] = get_robot_name(int(row["subject"]))
        else:
            barcode_to_var_name[row["barcode"]] = f"L{int(row['subject'])}"

    all_measurement_df = pd.DataFrame()

    # Load the measurement data and concat it to all_measurement_df
    for robot_idx in range(1, 6):
        robot_name = get_robot_name(robot_idx)

        measurement_fname = f"{data_dir}/Robot{robot_idx}_Measurement.dat"
        measurement_df = parse_whitespace_file(measurement_fname)
        assert (
            measurement_df.shape[1] == 4
        ), f"Expected 4 columns in {measurement_fname}"
        measurement_df.columns = ["timestamp", "barcode", "range", "bearing"]

        logger.info(
            "Loading %d range-bearing measurements for robot %s",
            measurement_df.shape[0],
            robot_name,
        )

        # Convert barcode to robot or landmark name
        # Filter out measurements with unknown barcodes
        valid_meas = measurement_df["barcode"].isin(barcode_to_var_name.keys())
        measurement_df = measurement_df[valid_meas]
        measurement_df["measured_var_name"] = measurement_df["barcode"].apply(
            lambda x: barcode_to_var_name[x]
        )
        if (~valid_meas).any():
            logger.warning(
                "Found %d measurements with unknown barcodes",
                (~valid_meas).sum(),
            )

        measurement_df["is_robot"] = measurement_df["measured_var_name"].apply(
            lambda x: "L" not in x
        )
        measurement_df["robot_var_name"] = robot_name

        all_measurement_df = pd.concat([all_measurement_df, measurement_df])

    # Sort by timestamp
    all_measurement_df.sort_values(by="timestamp", inplace=True)
    return all_measurement_df


def get_all_odoms(data_dir: str):
    """
    Returns all odometry measurements from all robots in ascending timestamp
    """
    all_odoms = {}
    for robot_idx in range(1, 6):
        robot_name: str = get_robot_name(robot_idx)

        odom_fname = f"{data_dir}/Robot{robot_idx}_Odometry.dat"
        odom_df = parse_whitespace_file(odom_fname)
        assert odom_df.shape[1] == 3, f"Expected 3 columns in {odom_fname}"
        odom_df.columns = ["timestamp", "v", "w"]

        logger.info(
            "Loading %d odometry measurements for robot %s",
            odom_df.shape[0],
            robot_name,
        )
        odom_df.sort_values(by="timestamp", inplace=True)
        all_odoms[robot_name] = odom_df.to_numpy(np.float64)
    return all_odoms


def integrate_odom(odoms: np.ndarray):
    """
    Given odometry measurements in the form of (timestamp, v, w), integrate the odometry

    Returns a numpy array of (timestamp, x, y, theta)
    """
    assert odoms.shape[1] == 3, f"Expected 3 columns in {odoms}"
    integrated_odom = np.zeros((odoms.shape[0], 4))
    integrated_odom[:, 0] = odoms[:, 0]
    for i in range(1, integrated_odom.shape[0]):
        dt = integrated_odom[i, 0] - integrated_odom[i - 1, 0]
        integrated_odom[i, 1] = integrated_odom[i - 1, 1] + dt * odoms[
            i - 1, 1
        ] * np.cos(integrated_odom[i - 1, 3])
        integrated_odom[i, 2] = integrated_odom[i - 1, 2] + dt * odoms[
            i - 1, 1
        ] * np.sin(integrated_odom[i - 1, 3])
        integrated_odom[i, 3] = integrated_odom[i - 1, 3] + dt * odoms[i - 1, 2]
    return integrated_odom


def parse_whitespace_file(filepath: str) -> pd.DataFrame:
    """
    Load the file into a dataframe with whitespace separator
    Ignores lines starting with #
    Checks for NaNs
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        comment="#",
    )

    assert df.notna().all().all(), f"NaNs found in {filepath}"
    return df


def parse_data(
    dirpath: str,
    start_time: float,
    end_time: float,
    range_translation_stddev=0.1,
    translation_stddev_rate=0.01,
    rotation_stddev_rate=0.01,
    landmark_stddev=0.1,
    add_landmark_prior: bool = False,
) -> FactorGraphData:
    """
    translation_stddev_rate and rotation_stddev_rate: stddev = rate * time difference
    The pose-landmark measurement and range measurement uses range_translation_stddev
    """
    fg = FactorGraphData(dimension=2)

    add_landmarks(fg, dirpath, landmark_stddev, add_landmark_prior)

    # Load everything
    all_measurements = get_all_measurements(
        os.path.join(dirpath, "Barcodes.dat"), dirpath
    )
    all_odoms = get_all_odoms(dirpath)
    all_gts = dict(
        [
            (
                get_robot_name(robot_idx),
                load_robot_gt(f"{dirpath}/Robot{robot_idx}_Groundtruth.dat"),
            )
            for robot_idx in range(1, 6)
        ]
    )

    # Integrate odom and store it as odom_poses, we will use this to interpolate odom measurements
    # between any two timestamps
    all_odom_poses: dict[str, Trajectory] = dict()
    for robot_name, odoms in all_odoms.items():
        all_odom_poses[robot_name] = Trajectory(integrate_odom(odoms))

    # Filter measurements such that:
    # 1. The measurement takes place between the first and last ground truth measurement
    # 2. The measurement takes place between the first and last odom measurement
    for robot_name, gt in all_gts.items():
        odom_poses = all_odom_poses[robot_name]
        adjusted_start_time = max(start_time, gt.min_ts, odom_poses.min_ts)
        adjusted_end_time = min(end_time, gt.max_ts, odom_poses.max_ts)

        # Valid measurements are either: not involving the robot, or within the time period of the ground truth
        valid_measurements = (all_measurements["robot_var_name"] != robot_name) & (
            all_measurements["measured_var_name"] != robot_name
        ) | (
            all_measurements["timestamp"].between(
                adjusted_start_time, adjusted_end_time
            )
        )
        all_measurements = all_measurements[valid_measurements]
        logger.info(
            f"Filtered out {(~valid_measurements).sum()} range-bearing measurements,"
        )

    # Maps robot name and timestamp to a pose number
    pose_ts_to_num: dict[str, dict[float, int]] = dict(
        [(name, {}) for name in all_odoms]
    )
    var_name_counter = dict([(name, 0) for name in all_odoms])

    logger.info("Adding range-bearing measurements")
    for _, row in tqdm(
        valid_measurements.iterrows(), total=valid_measurements.shape[0]
    ):
        timestamp = row["timestamp"]
        robot_var_name = row["robot_var_name"]
        measured_var_name = row["measured_var_name"]
        is_robot = row["is_robot"]
        range_meas = row["range"]
        bearing_meas = row["bearing"]

        robot_names = [robot_var_name]
        if is_robot:
            robot_names.append(measured_var_name)
        # Add a new pose variable if it does not exist
        for robot_name in robot_names:
            if timestamp not in pose_ts_to_num[robot_name]:
                pose_var_name = f"{robot_name}{var_name_counter[robot_name]}"
                gt_pose, gt_dt = all_gts[robot_name].at_timestamp(timestamp)
                if gt_dt > 0.5:
                    logger.warning(
                        f"Large time difference of {gt_dt:.2f} between odometry and ground"
                        f"truth at timestamp {timestamp:.2f} for robot {robot_name}",
                    )
                pose_var = PoseVariable2D(
                    pose_var_name,
                    gt_pose[:2],
                    gt_pose[2],
                    timestamp,
                )
                fg.add_pose_variable(pose_var)
                pose_ts_to_num[robot_name][timestamp] = var_name_counter[robot_name]
                var_name_counter[robot_name] += 1

        # Variable pair, either pose-pose or pose-landmark
        # var_name_counter at this point is always the number of variables + 1
        # The latest variable is always the one corresponding to the current measurement
        # since the measurements are in ascending timestamp

        association = (
            f"{robot_var_name}{var_name_counter[robot_var_name] - 1}",
            f"{measured_var_name}{var_name_counter[measured_var_name] - 1}"
            if is_robot
            else measured_var_name,
        )

        # Add measurement for pose-landmark as a translation vector and pose-pose as range-only
        # This is done because PyFg does not support pose-pose without rotation measurements
        if is_robot:
            measurement = FGRangeMeasurement(
                association, range_meas, range_translation_stddev, timestamp
            )
            fg.add_range_measurement(measurement)
        else:
            # Add range-bearing measurement as a Pose-landmark measurement
            x, y = range_meas * np.cos(bearing_meas), range_meas * np.sin(bearing_meas)
            measurement = PoseToLandmarkMeasurement2D(
                association[0],
                association[1],
                x,
                y,
                range_translation_stddev,
                timestamp,
            )
            fg.add_pose_landmark_measurement(measurement)

    # Use interpolated odometry poses to create odometry factors between pose variables
    logger.info(f"Adding odometry for {len(pose_ts_to_num)} pose variables")
    for robot_name, timestamp_to_num in pose_ts_to_num.items():
        odom_poses = all_odom_poses[robot_name]
        # Convert to list of tuples for easy iteration
        timestamp_to_num = np.array(list(timestamp_to_num.items()))
        assert (timestamp_to_num[1:, 0] - timestamp_to_num[:-1, 0] > 0).all(), (
            f"Timestamps are not in ascending order for robot {robot_name}: "
            f"{timestamp_to_num[:, 0]}"
        )

        # Interpolated odom
        for i in range(1, timestamp_to_num.shape[0]):
            prev_ts, prev_num = timestamp_to_num[i - 1]
            curr_ts, curr_num = timestamp_to_num[i]
            # Interpolate between the two poses
            prev_pose, _ = odom_poses.at_timestamp(prev_ts)
            curr_pose, _ = odom_poses.at_timestamp(curr_ts)
            dt = curr_ts - prev_ts
            delta_pose = curr_pose - prev_pose

            # Add odometry factor
            odom_factor = PoseMeasurement2D(
                f"{robot_name}{prev_num}",
                f"{robot_name}{curr_num}",
                delta_pose[0],
                delta_pose[1],
                delta_pose[2],
                translation_stddev_rate * dt,
                rotation_stddev_rate * dt,
                curr_ts,
            )
            fg.add_pose_measurement(odom_factor)

    return fg


"""
MRCLAM Dataset Splitting:

Due to missing ground truth for certain robots, we recommend the following splits

Dataset 3A: [0, 1248294983.360]
20 cm, 100 sec gap - robot C
Dataset 3B: [1248295100.130, end]

Dataset 5A: [0, 1248362946.495]
10 cm, 40 sec gap - robot D
Dataset 5B: [1248363027.950, 1248363991.000]
3 cm, 60 sec gap - robot B
Dataset 5C: [1248364052.743, end]


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str, help="path to the data directory")
    parser.add_argument(
        "save_path",
        default="mrclam_factor_graph.pyfg",
        type=str,
        help="path to save the data",
    )
    parser.add_argument(
        "--start_time", default=0, type=float, help="start time in seconds"
    )
    parser.add_argument(
        "--end_time", default=np.inf, type=float, help="end time in seconds"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the data")

    args = parser.parse_args()
    dirpath = args.dirpath
    assert os.path.exists(dirpath), f"Could not find directory at {dirpath}"

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.warning("Verbose mode on")
    logger.info("Parsing data from %s", dirpath)
    logger.error(
        "Storing robot to robot range bearing measurements as range-only measurements"
    )

    pyfg = parse_data(dirpath, args.start_time, args.end_time)
    pyfg.print_summary()

    if args.plot:
        pyfg.animate_odometry(show_gt=True, draw_range_lines=True)
    pyfg.save_to_file(args.save_path)
