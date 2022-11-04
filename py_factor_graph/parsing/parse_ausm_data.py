from os.path import expanduser, join
from os import listdir
from pathlib import Path
from typing import List, Tuple, List
import pandas as pd
import csv
from attrs import define, field

AUSM_DATA_DIR = join(expanduser("~"), "data", "ausm")
AUSM_EXPERIMENT_SUBDIRS = [
    join(AUSM_DATA_DIR, d)
    for d in listdir(AUSM_DATA_DIR)
    if Path(join(AUSM_DATA_DIR, d)).is_dir()
]
UWB_INDICATOR = 0
IMU_INDICATOR = 2
HEIGHT_INDICATOR = 3
DATA_INDICATOR = -1
EPSILON = 0.0001


@define
class ExperimentData:
    uwb_data: pd.DataFrame = field()
    imu_data: pd.DataFrame = field()
    height_data: pd.DataFrame = field()


@define
class CalibrationData:
    beacon_gt_pos: pd.DataFrame = field()
    calibration_coefficients: pd.DataFrame = field()


def _get_experiment_files_in_dir(experiment_dir) -> List[str]:
    return [
        join(experiment_dir, f)
        for f in listdir(experiment_dir)
        if Path(join(experiment_dir, f)).is_file() and "uwb_anchors" not in f
    ]


def _get_calibration_file_in_dir(experiment_dir) -> str:
    valid_files = [
        join(experiment_dir, f)
        for f in listdir(experiment_dir)
        if Path(join(experiment_dir, f)).is_file() and "uwb_anchors" in f
    ]
    assert len(valid_files) == 1
    return valid_files[0]


def _is_indicator_line(line: str, indicator: int) -> bool:
    try:
        return int(line.split(",")[0]) == indicator
    except ValueError:
        return False


def _is_uwb_indicator_line(line: str) -> bool:
    return _is_indicator_line(line, UWB_INDICATOR)


def _is_imu_indicator_line(line: str) -> bool:
    return _is_indicator_line(line, IMU_INDICATOR)


def _is_height_indicator_line(line: str) -> bool:
    return _is_indicator_line(line, HEIGHT_INDICATOR)


def _parse_experiment_file(experiment_file: str) -> ExperimentData:
    """Parses a file and returns the different measurements recorded in dataframes

    Args:
        experiment_file (str): the path to the file

    Returns:
        ExperimentData: the parsed data
    """
    print(f"Parsing experiment file: {experiment_file}")

    imu_lines = []
    imu_headers = ["time", "wx", "wy", "wz", "ax", "ay", "az", "qx", "qy", "qz", "qw"]
    add_imu_data = False

    uwb_lines = []
    uwb_headers = ["time", "module_id", "range", "self_range_error"]
    add_uwb_data = False

    height_lines = []
    height_headers = ["height"]
    add_height_data = False

    def _split_imu_data_line(line: str) -> List[float]:
        split_vals = line.split(",")
        assert len(split_vals) == len(imu_headers)
        return [float(v) for v in split_vals]

    def _split_uwb_data_line(line: str) -> List[float]:
        split_vals = line.split(",")
        assert len(split_vals) == len(uwb_headers), f"Unexpected UWB data line: {line}"
        return [float(v) for v in split_vals]

    def _split_height_data_line(line: str) -> List[float]:
        split_vals = line.split(",")
        assert len(split_vals) == len(height_headers)
        return [float(v) for v in split_vals]

    # iterate over the lines
    with open(experiment_file, "r") as f:
        for line in f:
            # make sure that at most one of the flags is true
            assert sum([add_imu_data, add_uwb_data, add_height_data]) <= 1
            assert not line.startswith("-1"), f"Invalid line: {line}"

            if add_imu_data:
                imu_lines.append(_split_imu_data_line(line))
                add_imu_data = False
            elif add_uwb_data:
                uwb_lines.append(_split_uwb_data_line(line))
                add_uwb_data = False
            elif add_height_data:
                height_lines.append(_split_height_data_line(line))
                add_height_data = False
            elif _is_imu_indicator_line(line):
                add_imu_data = True
            elif _is_uwb_indicator_line(line):
                add_uwb_data = True
            elif _is_height_indicator_line(line):
                add_height_data = True
            else:
                raise ValueError(f"Unknown line: {line}")

    # make dataframes from the lines
    imu_data = pd.DataFrame(imu_lines, columns=imu_headers, dtype=float)
    uwb_data = pd.DataFrame(uwb_lines, columns=uwb_headers, dtype=float)
    height_data = pd.DataFrame(height_lines, columns=height_headers, dtype=float)

    return ExperimentData(uwb_data, imu_data, height_data)


def _parse_calibration_file(calibration_file: str) -> CalibrationData:
    """parses the calibration file and returns the calibration data for the
    experimental setups

    # TODO: right now this code is a bit of a mess, clean it up later

    Args:
        calibration_file (str): path to the calibration file

    Returns:
        CalibrationData: the calibration data
    """
    print(f"Parsing calibration file: {calibration_file}")

    beacon_header = "a,X,Y,Z"
    coeff_header = "[],'A0','B0','A1','B1','A2','B2','A3','B3'"

    def _is_beacon_gt_header(line: str) -> bool:
        return beacon_header in line

    def _is_calibration_coeff_header(line: str) -> bool:
        return coeff_header in line

    def _is_other_header(line: str) -> bool:
        other_coeff_header = "'B0','B1','B2','B3'"
        building_bounds_header = "Building,,,,"
        uwb_raw_header = "UWB_Raw"
        valid_headers = [other_coeff_header, building_bounds_header, uwb_raw_header]
        return any([h in line for h in valid_headers])

    # these files are small - we'll iterate over the lines once to find the
    # starts/ends of different sections and then again to parse the data

    # first pass - find the starts of the different sections
    beacon_gt_start = None
    calibration_coeff_start = None
    other_headers_start = []
    with open(calibration_file, "r") as f:
        for i, line in enumerate(f):
            if _is_beacon_gt_header(line):
                assert beacon_gt_start is None
                beacon_gt_start = i
            elif _is_calibration_coeff_header(line):
                assert calibration_coeff_start is None
                calibration_coeff_start = i
            elif _is_other_header(line):
                other_headers_start.append(i)

    def _check_header_indices():
        assert (
            beacon_gt_start is not None
        ), f"Could not find beacon ground truth header in {calibration_file}"
        assert (
            calibration_coeff_start is not None
        ), f"Could not find calibration coefficients header in {calibration_file}"
        assert beacon_gt_start < calibration_coeff_start
        assert beacon_gt_start not in other_headers_start
        assert calibration_coeff_start not in other_headers_start

    _check_header_indices()

    def _parse_input_line(line: str) -> List:
        no_spaces = line.replace(" ", "").strip()
        no_quotes = no_spaces.replace('"', "").replace("'", "")
        return no_quotes.split(",")

    # second pass - parse the data
    beacon_gt_lines = []
    calibration_coeff_lines = []
    with open(calibration_file, "r") as f:
        reading_beacon_gt = False
        reading_calibration_coeff = False
        for i, line in enumerate(f):

            # set the reading flags
            if i == beacon_gt_start:
                reading_beacon_gt = True
                reading_calibration_coeff = False
            elif i == calibration_coeff_start:
                reading_beacon_gt = False
                reading_calibration_coeff = True
            elif i in other_headers_start:
                reading_beacon_gt = False
                reading_calibration_coeff = False

            # read the data
            if reading_beacon_gt:
                beacon_gt_lines.append(_parse_input_line(line))
            elif reading_calibration_coeff:
                calibration_coeff_lines.append(_parse_input_line(line))

    # make dataframes from the lines, header is the first line
    beacon_gt_data = pd.DataFrame(beacon_gt_lines[1:], columns=beacon_gt_lines[0])
    calibration_coeff_data = pd.DataFrame(
        calibration_coeff_lines[1:], columns=calibration_coeff_lines[0]
    )

    calibration_coeff_data = calibration_coeff_data.rename(columns={"[]": "scene"})
    beacon_gt_data = beacon_gt_data.rename(columns={"a": "beacon"})

    print(f"BGT:\n{beacon_gt_data}")
    print(f"CC:\n{calibration_coeff_data}")

    return CalibrationData(beacon_gt_data, calibration_coeff_data)


if __name__ == "__main__":
    for sample_experiment_dir in AUSM_EXPERIMENT_SUBDIRS:
        sample_calibration_file = _get_calibration_file_in_dir(sample_experiment_dir)
        calibration_data = _parse_calibration_file(sample_calibration_file)

        for sample_experiment_file in _get_experiment_files_in_dir(
            sample_experiment_dir
        ):
            experiment_data = _parse_experiment_file(sample_experiment_file)
