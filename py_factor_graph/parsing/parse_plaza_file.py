"""
For parsing the Plaza dataset

paper: https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20311
dataset: https://infoscience.epfl.ch/record/283435
"""
from typing import List, Dict, Tuple, Callable
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
from attrs import define, field

ODOM_EXTENSION = "_DR.csv"
ODOM_PATH_EXTENSION = "_DRp.csv"
GT_ROBOT_EXTENSION = "_GT.csv"
DIST_MEASURE_EXTENSION = "_TD.csv"
GT_LANDMARK_EXTENSION = "_TL.csv"


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
        headers = ["time", "beacon_id", "x", "y"]
        return pd.read_csv(self.gt_landmark_file, names=headers)

    def get_beacon_id_to_idx_mapping(self) -> Dict[int, int]:
        beacon_id_to_idx = {}
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


def _find_nearest_time_index(df: pd.DataFrame, time: float) -> int:
    return np.argmin(np.abs(df["time"] - time))


def _get_list_of_range_measures(
    data_files: PlazaDataFiles,
) -> List[Tuple[float, str, str, float]]:
    beacon_id_to_idx = data_files.get_beacon_id_to_idx_mapping()
    gt_pose_df = data_files.robot_gt_df()
    range_df = data_files.dist_measure_df()
    range_df["beacon_id"] = range_df["beacon_id"].apply(lambda x: beacon_id_to_idx[x])

    range_measure_list: List[Tuple[str, str, float]] = []
    for _, row in range_df.iterrows():
        range_measure_time = row["time"]
        nearest_robot_pose_idx = _find_nearest_time_index(
            gt_pose_df, range_measure_time
        )
        robot_pose_name = f"A{nearest_robot_pose_idx}"
        beacon_pose_name = f"L{int(row['beacon_id'])}"
        measured_distance = row["distance"]
        range_measure_list.append(
            (range_measure_time, robot_pose_name, beacon_pose_name, measured_distance)
        )

    return range_measure_list


def _obtain_calibrations_for_radios(data_files: PlazaDataFiles) -> Dict[int, Callable]:
    range_measures = _get_list_of_range_measures(data_files)
    gt_pose_df = data_files.robot_gt_df()
    beacon_idxs = data_files.get_beacon_id_to_idx_mapping().values()

    # group the range measures by beacon and pair them with the true range
    calibration_pairs = {x: [] for x in beacon_idxs}
    for measure in range_measures:
        meas_time, robot_name, beacon_name, measured_distance = measure
        robot_idx = int(robot_name[1:])
        beacon_idx = int(beacon_name[1:])

        true_robot_location = gt_pose_df.iloc[robot_idx][["x", "y"]].values
        true_beacon_location = data_files.landmark_gt_df().iloc[beacon_idx][
            ["x", "y"]
        ].values

        true_range = np.linalg.norm(true_robot_location - true_beacon_location)
        
        calibration_pairs[beacon_idx].append((true_range, measured_distance))

    # for each beacon we now have a list of true range, measured range pairs
    # we will fit a linear model to these pairs and use that as the calibration
    calibrations = {}
    for beacon_idx, pairs in calibration_pairs.items():
        true_ranges = np.array([x[0] for x in pairs])
        measured_ranges = np.array([x[1] for x in pairs])

        # we want a mapping from measured range to true range
        slope, intercept, _, _, _ = linregress(measured_ranges, true_ranges)
        calibrations[beacon_idx] = lambda x: slope * x + intercept

    return calibrations


def _add_range_measurements(fg: FactorGraphData, data_files: PlazaDataFiles):
    beacon_id_to_idx = data_files.get_beacon_id_to_idx_mapping()
    range_df = data_files.dist_measure_df()
    gt_pose_df = data_files.robot_gt_df()

    calibrations = _obtain_calibrations_for_radios(data_files)
    range_measures = _get_list_of_range_measures(data_files)

    range_stddev = 3.0
    for measure in range_measures:
        meas_time, pose_name, beacon_name, measured_distance = measure
        beacon_idx = int(beacon_name[1:])
        calibrated_distance = calibrations[beacon_idx](measured_distance)
        range_measure = FGRangeMeasurement(
            association=(pose_name, beacon_name),
            dist=calibrated_distance,
            stddev=range_stddev,
            timestamp=meas_time,
        )
        fg.add_range_measurement(range_measure)



def parse_plaza_files(dirpath: str) -> FactorGraphData:
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
    _add_range_measurements(fg, data_files)

    return fg


if __name__ == "__main__":
    data_dir = "/home/alan/experimental_data/plaza/Plaza1"
    fg = parse_plaza_files(data_dir)
    fg.print_summary()
