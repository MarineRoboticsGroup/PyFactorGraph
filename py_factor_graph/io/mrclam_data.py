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
from py_factor_graph.measurements import FGRangeMeasurement, PoseMeasurement2D
from py_factor_graph.priors import LandmarkPrior2D
from py_factor_graph.utils.name_utils import get_robot_char_from_number
from py_factor_graph.variables import LandmarkVariable2D, PoseVariable2D

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

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Gets a df with timestamp, x, y, theta and creates a trajectory object
        """
        self.data = df.to_numpy()
        self.min_ts = np.min(self.data[:, 0])
        self.max_ts = np.max(self.data[:, 0])
        assert self.data.shape[1] == 4, f"Expected 4 columns in {df}"

        # Sort by timestamp
        self.data = self.data[self.data[:, 0].argsort()]

    def at_timestamp(self, timestamp: float) -> Tuple[np.ndarray, float]:
        """
        Returns the pose at the given timestamp by interpolating between the closest two poses
          and the change in time from the closest timestamp
        """
        # Find the closest timestamp
        if timestamp < self.min_ts:
            logger.warning(
                f"Timestamp {timestamp:.3f} is before the first timestamp {self.min_ts:.3f}, returning 0"
            )
            return (np.zeros(3), 0)

        if timestamp > self.max_ts:
            logger.warning(
                f"Timestamp {timestamp:.3f} is after the last timestamp {self.max_ts:.3f}, returning extrapolated pose"
            )

        closest_timestamp_index = np.searchsorted(
            self.data[:, 0], timestamp, side="left"
        )
        if closest_timestamp_index == 0:
            return (np.zeros(3), 0)

        closest_poses = self.data[
            closest_timestamp_index - 1 : closest_timestamp_index + 1, 1:
        ]
        closest_timestamps = self.data[
            closest_timestamp_index - 1 : closest_timestamp_index + 1, 0
        ]

        interpolated_pose = closest_poses[0] + (closest_poses[1] - closest_poses[0]) * (
            (timestamp - closest_timestamps[0])
            / (closest_timestamps[1] - closest_timestamps[0])
        )

        return (interpolated_pose, np.abs(timestamp - closest_timestamps[0]))


def get_robot_name(robot_num: int):
    """
    Convert robot number to A-E
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
    return Trajectory(df)


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

    # Add all landmarks as vars, and add a prior if requested
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
        all_odoms[robot_name] = odom_df.to_numpy(np.float32)

    return all_odoms


def inject_new_odoms(all_odoms, all_measurements):
    # Inject an odometry measurement at every measurement so that the odoms and measurements are synchronized
    new_odoms = dict([(name, []) for name in all_odoms.keys()])

    logger.info("Synchronizing odometry and range-bearing measurements")
    for _, row in tqdm(all_measurements.iterrows(), total=all_measurements.shape[0]):
        timestamp = row["timestamp"]
        robot_var_names = [row["robot_var_name"]]
        if row["is_robot"]:
            robot_var_names.append(row["measured_var_name"])

        # Add an odometry measurement for each robot
        for robot_var_name in robot_var_names:
            # Find the closest odometry measurement that is before the timestamp
            robot_odoms = all_odoms[robot_var_name]
            closes_odom_idx = np.searchsorted(robot_odoms[:, 0], timestamp, side="left")
            if closes_odom_idx == 0:
                logging.warning(
                    "No odometry measurement found before timestamp %f for robot %s",
                    timestamp,
                    robot_var_name,
                )
                new_odoms[robot_var_name].append([timestamp, 0, 0])
            else:
                closest_odom = robot_odoms[closes_odom_idx - 1]
                new_odoms[robot_var_name].append(
                    [timestamp, closest_odom[1], closest_odom[2]]
                )

    return new_odoms


def parse_whitespace_file(filepath: str) -> pd.DataFrame:
    # Load the file into a dataframe with whitespace separator
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
    range_stddev=0.3,
    bearing_stddev=0.3,
    translation_stddev_rate=0.01,
    rotation_stddev_rate=0.01,
    landmark_stddev=0.1,
    add_landmark_prior: bool = False,
) -> FactorGraphData:
    """
    translation_stddev_rate and rotation_stddev_rate: stddev = rate * time difference
    """
    fg = FactorGraphData(dimension=2)

    add_landmarks(fg, dirpath, landmark_stddev, add_landmark_prior)

    # Get measurements and odometry for all robots
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

    new_odoms = inject_new_odoms(all_odoms, all_measurements)
    for robot_name, odoms in new_odoms.items():
        new_odoms[robot_name] = np.unique(np.array(odoms), axis=0)  # remove duplicates
        all_odoms[robot_name] = np.vstack(
            (all_odoms[robot_name], new_odoms[robot_name])
        )
        # Sort all_odoms by timestamp, which is the first column
        all_odoms[robot_name] = all_odoms[robot_name][
            np.argsort(all_odoms[robot_name][:, 0])
        ]

    # Add all measurements as pose variables and range-bearing factors

    # Maps robot name and timestamp to a pose variable name
    pose_vars = dict()
    var_name_counter = dict([(name, 0) for name in all_odoms.keys()])

    logger.info("Adding range-bearing measurements")
    for _, row in tqdm(all_measurements.iterrows(), total=all_measurements.shape[0]):
        timestamp = row["timestamp"]
        robot_var_name = row["robot_var_name"]
        measured_var_name = row["measured_var_name"]
        is_robot = row["is_robot"]
        range_meas = row["range"]
        bearing_meas = row["bearing"]

        robot_names = [robot_var_name]
        if is_robot:
            robot_names.append(measured_var_name)

        for robot_name in robot_names:
            if (robot_name, timestamp) not in pose_vars:
                # Add a new pose variable
                pose_var_name = f"{robot_name}{var_name_counter[robot_name]}"
                gt_pose, gt_dt = all_gts[robot_name].at_timestamp(timestamp)
                if gt_dt > 0.1:
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
                pose_vars[(robot_name, timestamp)] = pose_var
                var_name_counter[robot_name] += 1

        # Variable pair, either pose-pose or pose-landmark
        association = (
            f"{robot_var_name}{var_name_counter[robot_var_name] - 1}",
            f"{measured_var_name}{var_name_counter[measured_var_name] - 1}"
            if is_robot
            else measured_var_name,
        )
        # Add range and bearing measurements
        range_meas = FGRangeMeasurement(
            association,
            range_meas,
            range_stddev,
            timestamp,
        )
        fg.add_range_measurement(range_meas)

    # Integrate along the odometry chain and add odometry measurements between two pose variables
    # Since odometry comes in at a much faster rate than measurements, this integration
    # reduces the number of variables by an order of magnitude

    for robot_idx in range(1, 6):
        robot_name = get_robot_name(robot_idx)
        logger.info(f"Adding odometry measurements for Robot {robot_name}")
        odoms = all_odoms[robot_name]

        # Cur_transform is in the frame of the previous pose variable
        cur_transform = np.zeros(4)  # ts, x, y, theta
        # Prev_odom in the frame of the current_transform, this is the velocity at cur_transform's timestamp
        cur_odom = np.zeros(3)  # ts, v, w

        for odom in tqdm(odoms):
            timestamp, v, w = odom
            # Perform integration
            dt = timestamp - cur_transform[0]
            cur_transform[1] += dt * cur_odom[1] * np.cos(cur_transform[3])
            cur_transform[2] += dt * cur_odom[1] * np.sin(cur_transform[3])
            cur_transform[3] += dt * cur_odom[2]
            cur_odom = odom

            if (robot_name, timestamp) in pose_vars:
                pose_var = pose_vars[(robot_name, timestamp)].name
                pose_num = int(pose_var[1:])
                if pose_num > 0:
                    # Add odom factor between X[i] and X[i-1]
                    prev_pose_var = f"{robot_name}{pose_num - 1}"
                    # time difference between the current and previous pose variable
                    # used to calculate the stddev of the odometry measurement
                    transform_dt = timestamp - cur_transform[0]
                    # stddev = rate * time difference
                    translation_stddev = transform_dt * translation_stddev_rate
                    rotation_stddev = transform_dt * rotation_stddev_rate

                    odom_meas = PoseMeasurement2D(
                        prev_pose_var,
                        pose_var,
                        cur_transform[1],
                        cur_transform[2],
                        cur_transform[3],
                        translation_stddev,
                        rotation_stddev,
                        timestamp,
                    )
                    fg.add_odom_measurement(robot_idx, odom_meas)

                    # Reset the current transform since it is in the frame of the previous pose variable
                    cur_transform = np.zeros(4)

    return fg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str, help="path to the data directory")
    parser.add_argument(
        "save_path",
        default="mrclam_factor_graph.pyfg",
        type=str,
        help="path to save the data",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the data")

    args = parser.parse_args()
    dirpath = args.dirpath
    assert os.path.exists(dirpath), f"Could not find file at {dirpath}"

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.warning("Verbose mode on")
    logger.info("Parsing data from %s", dirpath)

    pyfg = parse_data(dirpath)
    pyfg.print_summary()

    if args.plot:
        pyfg.animate_groundtruth()
        pyfg.animate_odometry(show_gt=True)

    pyfg.save_to_file(args.save_path)
