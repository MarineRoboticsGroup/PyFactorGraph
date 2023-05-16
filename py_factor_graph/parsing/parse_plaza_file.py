"""
For parsing the Plaza dataset

paper: https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20311
dataset: https://infoscience.epfl.ch/record/283435
"""
from typing import List, Dict, Tuple, Callable, Optional, Union
import copy
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    FGRangeMeasurement,
)
from py_factor_graph.priors import (
    PosePrior2D,
    PosePrior3D,
    LandmarkPrior2D,
    LandmarkPrior3D,
)
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.matrix_utils import (
    get_measurement_precisions_from_covariances,
)
from py_factor_graph.utils.name_utils import get_time_idx_from_frame_name
import matplotlib.pyplot as plt
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


@define
class UncalibratedRangeMeasurement:
    association: Tuple[str, str] = field()
    dist: float = field()
    timestamp: float = field()
    true_dist: Optional[float] = field(default=None)

    def set_true_dist(self, true_dist: float):
        self.true_dist = true_dist


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
    for idx, row in gt_pose_df.iterrows():
        pose_var = PoseVariable2D(
            name=f"A{idx}",
            true_position=(row["x"], row["y"]),
            true_theta=row["theta"],
            timestamp=row["time"],
        )
        fg.add_pose_variable(pose_var)


def _add_odometry_measurements(fg: FactorGraphData, data_files: PlazaDataFiles):
    odom_df = data_files.odom_df()

    translation_cov = (0.1) ** 2
    rot_cov = (0.01) ** 2
    trans_precision, rot_precision = get_measurement_precisions_from_covariances(
        translation_cov, rot_cov
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


@define
class LinearCalibrationModel:
    slope: float = field()
    intercept: float = field()

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.slope * x + self.intercept


def _get_residuals(
    uncalibrated_measurements: List[UncalibratedRangeMeasurement],
    linear_calibration: LinearCalibrationModel,
) -> np.ndarray:
    """
    We will fit a linear model to the range measurements and remove outliers.
    """
    measured_distances = np.array([x.dist for x in uncalibrated_measurements])
    true_distances = np.array([x.true_dist for x in uncalibrated_measurements])
    predicted_true_distances = linear_calibration(measured_distances)
    residuals = true_distances - predicted_true_distances
    return residuals


def _fit_linear_calibration_model(
    uncalibrated_measurements: List[UncalibratedRangeMeasurement],
) -> LinearCalibrationModel:
    """
    We will fit a linear model to the range measurements and remove outliers.
    """
    measured_dists = np.array([x.dist for x in uncalibrated_measurements])
    true_dists = np.array([x.true_dist for x in uncalibrated_measurements])
    slope, intercept, r_value, p_value, std_err = linregress(measured_dists, true_dists)
    return LinearCalibrationModel(slope=slope, intercept=intercept)


def _apply_calibration_model(
    measurements: List[UncalibratedRangeMeasurement],
    calibration_model: LinearCalibrationModel,
    stddev: Optional[float] = None,
) -> List[FGRangeMeasurement]:
    # if we don't have a stddev, we will compute it from the residuals
    residuals = _get_residuals(measurements, calibration_model)
    calibrated_stddev = np.std(residuals)
    logger.debug(f"Calibrated stddev is {calibrated_stddev}")
    if stddev is None:
        stddev = calibrated_stddev

    logger.debug(f"Using stddev of {stddev} for range measurements.")
    calibrated_measurements: List[FGRangeMeasurement] = []
    for uncalibrated_measure in measurements:
        measured_dist = uncalibrated_measure.dist
        calibrated_dist = calibration_model(measured_dist)
        assert isinstance(calibrated_dist, float)
        calibrated_measure = FGRangeMeasurement(
            uncalibrated_measure.association,
            dist=calibrated_dist,
            stddev=stddev,
            timestamp=uncalibrated_measure.timestamp,
        )
        calibrated_measurements.append(calibrated_measure)

    return calibrated_measurements


def _get_inlier_set_of_range_measurements(
    uncalibrated_measurements: List[UncalibratedRangeMeasurement],
    inlier_stddev_threshold: float = 3.0,
    show_outlier_rejection: bool = False,
) -> List[UncalibratedRangeMeasurement]:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """

    def _plot_inliers_and_outliers(
        measurements: List[UncalibratedRangeMeasurement],
        outlier_mask: np.ndarray,
    ):
        inliers = [x for idx, x in enumerate(measurements) if idx not in outlier_mask]
        outliers = [x for idx, x in enumerate(measurements) if idx in outlier_mask]
        inlier_measured_dists = np.array([x.dist for x in inliers])
        inlier_true_dists = np.array([x.true_dist for x in inliers])
        outlier_measured_dists = np.array([x.dist for x in outliers])
        outlier_true_dists = np.array([x.true_dist for x in outliers])

        plt.scatter(
            inlier_measured_dists, inlier_true_dists, color="blue", label="inliers"
        )
        plt.scatter(
            outlier_measured_dists, outlier_true_dists, color="red", label="outliers"
        )
        plt.legend()
        plt.show(block=True)

    inliers_have_converged = False
    inlier_measurements = copy.deepcopy(uncalibrated_measurements)
    while not inliers_have_converged:
        # fit a linear model to the range measurements
        linear_calibration = _fit_linear_calibration_model(inlier_measurements)

        # compute the residuals and use them to find outliers
        residuals = _get_residuals(inlier_measurements, linear_calibration)
        res_stddev = np.std(residuals)
        outlier_mask = np.where(
            np.abs(residuals) > inlier_stddev_threshold * res_stddev
        )[0]

        # visualize the inliers and outliers
        if show_outlier_rejection:
            _plot_inliers_and_outliers(inlier_measurements, outlier_mask)

        # check if we have converged
        inliers_have_converged = len(outlier_mask) == 0
        if inliers_have_converged:
            break

        # remove any measurements that are outliers
        inlier_measurements = [
            x for idx, x in enumerate(inlier_measurements) if idx not in outlier_mask
        ]

    return inlier_measurements


def _obtain_calibrated_measurements(
    data_files: PlazaDataFiles,
    range_measures: List[UncalibratedRangeMeasurement],
    stddev: Optional[float] = None,
) -> List[FGRangeMeasurement]:
    gt_pose_df = data_files.robot_gt_df()
    beacon_idxs = data_files.get_beacon_id_to_idx_mapping().values()
    beacon_gt_df = data_files.landmark_gt_df()

    # group the range measures by beacon and add the true range (from GPS) to each
    calibration_pairs: Dict[int, List[UncalibratedRangeMeasurement]] = {
        x: [] for x in beacon_idxs
    }
    for measure in range_measures:
        pose_name, beacon_name = measure.association
        robot_idx = int(pose_name[1:])
        beacon_idx = int(beacon_name[1:])

        true_robot_location = gt_pose_df.iloc[robot_idx][["x", "y"]].values
        true_beacon_location = beacon_gt_df.iloc[beacon_idx][["x", "y"]].values
        true_range = float(np.linalg.norm(true_robot_location - true_beacon_location))

        measure.set_true_dist(true_range)
        calibration_pairs[beacon_idx].append(measure)

    inlier_measurements: Dict[int, List[UncalibratedRangeMeasurement]] = {}
    for beacon_idx, measures in calibration_pairs.items():
        inlier_measurements[beacon_idx] = _get_inlier_set_of_range_measurements(
            measures
        )

    all_calibrated_measurements: List[FGRangeMeasurement] = []
    for beacon_idx, measures in inlier_measurements.items():
        linear_calibration = _fit_linear_calibration_model(measures)
        calibrated_measurements = _apply_calibration_model(
            measures, linear_calibration, stddev=stddev
        )
        all_calibrated_measurements.extend(calibrated_measurements)

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
            pause=0.0001,
            num_range_measures_shown=4,
            clear_ranges_every_frame=True,
        )

    # save the factor graph to file
    save_path = os.path.expanduser(
        "~/experimental_data/plaza/Plaza1/factor_graph.pickle"
    )
    fg.save_to_file(save_path)
