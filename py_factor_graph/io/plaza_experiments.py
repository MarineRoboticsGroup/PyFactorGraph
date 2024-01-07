"""
For parsing the Plaza dataset

paper: https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20311
dataset: https://infoscience.epfl.ch/record/283435
"""
from typing import List, Dict, Tuple, Optional
import os
import numpy as np
import pandas as pd

from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    FGRangeMeasurement,
)
from py_factor_graph.calibrations.range_measurement_calibration import (
    UncalibratedRangeMeasurement,
    get_inlier_set_of_range_measurements,
    get_linearly_calibrated_measurements,
)
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.matrix_utils import (
    get_measurement_precisions_from_covariances,
)
from attrs import define, field

ODOM_EXTENSION = "_DR.csv"
ODOM_PATH_EXTENSION = "_DRp.csv"
GT_ROBOT_EXTENSION = "_GT.csv"
DIST_MEASURE_EXTENSION = "_TD.csv"
GT_LANDMARK_EXTENSION = "_TL.csv"

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


def _get_file_with_extension(files: List[str], extension: str) -> str:
    candidate_files = [f for f in files if f.endswith(extension)]
    assert (
        len(candidate_files) == 1
    ), f"Found {len(candidate_files)} files with extension {extension} but expected 1."
    assert os.path.isfile(
        candidate_files[0]
    ), f"Selected {candidate_files[0]} but it is not a valid file."
    return candidate_files[0]


@define
class PlazaDataFiles:
    dirpath: str = field()

    # have all of the files but construct after init
    odom_file: str = field(init=False)
    odom_path_file: str = field(init=False)
    gt_robot_file: str = field(init=False)
    dist_measure_file: str = field(init=False)
    gt_landmark_file: str = field(init=False)

    @dirpath.validator
    def _check_dirpath(self, attribute, value):
        if not os.path.isdir(value):
            raise ValueError(f"dirpath {value} is not a directory.")

    def __attrs_post_init__(self):
        all_files = [os.path.join(self.dirpath, f) for f in os.listdir(self.dirpath)]
        self.odom_file = _get_file_with_extension(all_files, ODOM_EXTENSION)
        self.odom_path_file = _get_file_with_extension(all_files, ODOM_PATH_EXTENSION)
        self.gt_robot_file = _get_file_with_extension(all_files, GT_ROBOT_EXTENSION)
        self.dist_measure_file = _get_file_with_extension(
            all_files, DIST_MEASURE_EXTENSION
        )
        self.gt_landmark_file = _get_file_with_extension(
            all_files, GT_LANDMARK_EXTENSION
        )

    def robot_gt_df(self) -> pd.DataFrame:
        headers = ["time", "x", "y", "theta"]
        return pd.read_csv(self.gt_robot_file, names=headers)

    def odom_df(self) -> pd.DataFrame:
        headers = ["time", "dx", "dtheta"]
        return pd.read_csv(self.odom_file, names=headers)

    def odom_path_df(self) -> pd.DataFrame:
        headers = ["time", "x", "y", "theta"]
        return pd.read_csv(self.odom_path_file, names=headers)

    def dist_measure_df(self) -> pd.DataFrame:
        headers = ["time", "robot_id", "beacon_id", "distance"]
        range_df = pd.read_csv(self.dist_measure_file, names=headers)
        assert (
            len(range_df["robot_id"].unique()) == 1
        ), "Multiple robot ids found in range file."
        return range_df

    def landmark_gt_df(self) -> pd.DataFrame:
        headers = ["beacon_id", "x", "y"]
        return pd.read_csv(self.gt_landmark_file, names=headers)

    def get_beacon_id_to_idx_mapping(self) -> Dict[int, int]:
        beacon_id_to_idx: Dict[int, int] = {}
        with open(self.gt_landmark_file, "r") as f:
            for line in f.readlines():
                beacon_id, x, y = line.split(",")
                beacon_id_to_idx[int(beacon_id)] = len(beacon_id_to_idx)
        return beacon_id_to_idx


def _set_beacon_variables(fg: FactorGraphData, data_files: PlazaDataFiles):
    beacon_id_to_idx = data_files.get_beacon_id_to_idx_mapping()
    with open(data_files.gt_landmark_file, "r") as f:
        for line in f.readlines():
            beacon_id, x, y = line.split(",")
            beacon_idx = beacon_id_to_idx[int(beacon_id)]
            beacon_var = LandmarkVariable2D(
                name=f"L{beacon_idx}",
                true_position=(float(x), float(y)),
            )
            fg.add_landmark_variable(beacon_var)


def _set_pose_variables(fg: FactorGraphData, data_files: PlazaDataFiles):
    gt_pose_df = data_files.robot_gt_df()
    if "plaza2" in data_files.dirpath.lower():
        logger.warning("Plaza2 data detected. Adding pi offset to theta.")
        theta_offset = np.pi
    else:
        theta_offset = 0.0
    for idx, row in gt_pose_df.iterrows():
        pose_var = PoseVariable2D(
            name=f"A{idx}",
            true_position=(row["x"], row["y"]),
            true_theta=row["theta"] + theta_offset,
            timestamp=row["time"],
        )
        fg.add_pose_variable(pose_var)


def _add_odometry_measurements(fg: FactorGraphData, data_files: PlazaDataFiles):
    odom_df = data_files.odom_df()

    translation_cov = (0.1) ** 2
    rot_cov = (0.01) ** 2
    trans_precision, rot_precision = get_measurement_precisions_from_covariances(
        translation_cov, rot_cov, mat_dim=3
    )

    for idx, row in odom_df.iterrows():
        odom_measure = PoseMeasurement2D(
            base_pose=f"A{idx}",
            to_pose=f"A{idx+1}",
            x=row["dx"],
            y=0.0,
            theta=row["dtheta"],
            translation_precision=trans_precision,
            rotation_precision=rot_precision,
            timestamp=row["time"],
        )
        fg.add_odom_measurement(robot_idx=0, odom_meas=odom_measure)


def _find_nearest_time_index(
    time_series: pd.Series, target_time: float, start_idx: int
) -> int:
    """
    We know that time is sorted and that we will be iterating through the
    time_series in order. As a result, we can start our search from the
    previous index we found.
    """
    for idx in range(start_idx, len(time_series)):
        if time_series[idx] >= target_time:
            prev_time = time_series[idx - 1]
            next_time = time_series[idx]
            prev_diff = abs(prev_time - target_time)
            next_diff = abs(next_time - target_time)
            return idx - 1 if prev_diff < next_diff else idx

    return len(time_series) - 1


def _parse_uncalibrated_range_measures(
    data_files: PlazaDataFiles,
) -> List[UncalibratedRangeMeasurement]:
    beacon_id_to_idx = data_files.get_beacon_id_to_idx_mapping()
    gt_pose_df = data_files.robot_gt_df()
    range_df = data_files.dist_measure_df()
    range_df["beacon_id"] = range_df["beacon_id"].apply(lambda x: beacon_id_to_idx[x])

    # collect a list of range measures for each robot-beacon pair - we will
    # average the measured distance and timestamps over these to get a single
    # range measurement for each robot-beacon pair
    range_measures: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    most_recent_pose_idx = 0
    for _, row in range_df.iterrows():
        range_measure_time = row["time"]
        nearest_robot_pose_idx = _find_nearest_time_index(
            gt_pose_df["time"], range_measure_time, most_recent_pose_idx
        )
        most_recent_pose_idx = nearest_robot_pose_idx
        robot_pose_name = f"A{nearest_robot_pose_idx}"
        beacon_pose_name = f"L{int(row['beacon_id'])}"
        measured_distance = row["distance"]

        association = (robot_pose_name, beacon_pose_name)
        if association not in range_measures:
            range_measures[association] = []
        range_measures[association].append((range_measure_time, measured_distance))

    range_measure_list: List[UncalibratedRangeMeasurement] = []
    for association, measures in range_measures.items():
        robot_pose_name, beacon_pose_name = association
        avg_measured_distance = float(np.mean([x[1] for x in measures]))
        measured_timestamp = float(np.mean([x[0] for x in measures]))
        range_measure_list.append(
            UncalibratedRangeMeasurement(
                association=association,
                dist=avg_measured_distance,
                timestamp=measured_timestamp,
            )
        )

    return range_measure_list


def _obtain_calibrated_measurements(
    data_files: PlazaDataFiles,
    uncalibrated_measures: List[UncalibratedRangeMeasurement],
    stddev: Optional[float] = None,
) -> List[FGRangeMeasurement]:
    gt_pose_df = data_files.robot_gt_df()
    beacon_idxs = data_files.get_beacon_id_to_idx_mapping().values()
    beacon_gt_df = data_files.landmark_gt_df()

    # group the range measures by beacon and add the true range (from GPS) to each
    calibration_pairs: Dict[int, List[UncalibratedRangeMeasurement]] = {
        x: [] for x in beacon_idxs
    }
    for uncal_measure in uncalibrated_measures:
        pose_name, beacon_name = uncal_measure.association
        robot_idx = int(pose_name[1:])
        beacon_idx = int(beacon_name[1:])

        true_robot_location = gt_pose_df.iloc[robot_idx][["x", "y"]].values
        true_beacon_location = beacon_gt_df.iloc[beacon_idx][["x", "y"]].values
        true_range = float(np.linalg.norm(true_robot_location - true_beacon_location))

        uncal_measure.set_true_dist(true_range)
        calibration_pairs[beacon_idx].append(uncal_measure)

    inlier_measurements: Dict[int, List[UncalibratedRangeMeasurement]] = {
        beacon_idx: get_inlier_set_of_range_measurements(measures)
        for beacon_idx, measures in calibration_pairs.items()
    }

    all_calibrated_measurements: List[FGRangeMeasurement] = []
    for beacon_idx, measures in inlier_measurements.items():
        calibrated_measurements = get_linearly_calibrated_measurements(measures)
        all_calibrated_measurements.extend(calibrated_measurements)

    if stddev is not None:
        for measure in all_calibrated_measurements:
            measure.stddev = stddev

    return all_calibrated_measurements


def _add_range_measurements(
    fg: FactorGraphData,
    data_files: PlazaDataFiles,
    range_stddev: Optional[float] = None,
):
    uncalibrated_range_measures = _parse_uncalibrated_range_measures(data_files)
    calibrated_ranges = _obtain_calibrated_measurements(
        data_files, uncalibrated_range_measures, stddev=range_stddev
    )
    for range_measure in calibrated_ranges:
        fg.add_range_measurement(range_measure)


def parse_plaza_files(
    dirpath: str, range_stddev: Optional[float] = None
) -> FactorGraphData:
    data_files = PlazaDataFiles(dirpath)
    if "gesling" in dirpath.lower():
        raise NotImplementedError(
            """
            Gesling data not yet supported. This data requires some
            additional calibration, as there are multiple radios attached to the robot
            (https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20311)
            """
        )

    fg = FactorGraphData(dimension=2)
    _set_pose_variables(fg, data_files)
    _add_odometry_measurements(fg, data_files)
    _set_beacon_variables(fg, data_files)
    _add_range_measurements(fg, data_files, range_stddev=range_stddev)

    return fg


if __name__ == "__main__":
    import os

    data_dir = os.path.expanduser("~/experimental_data/plaza/Plaza1")

    # parse and print summary
    fg = parse_plaza_files(data_dir)
    fg.print_summary()

    # animate if desired
    visualize = False
    if visualize:
        fg.animate_odometry(
            show_gt=True,
            pause_interval=0.01,
            draw_range_lines=True,
            draw_range_circles=False,
            num_timesteps_keep_ranges=1,
        )

    # save the factor graph to file
    save_path = os.path.expanduser(
        "~/experimental_data/plaza/Plaza1/factor_graph.pickle"
    )
    fg.save_to_file(save_path)
