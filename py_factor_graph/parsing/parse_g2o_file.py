from typing import List, Any, Union, Tuple, Optional
from os.path import isfile
import numpy as np

from py_factor_graph.variables import PoseVariable3D, LandmarkVariable2D
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    AmbiguousPoseMeasurement2D,
    PoseMeasurement3D,
    FGRangeMeasurement,
    AmbiguousFGRangeMeasurement,
)
from py_factor_graph.priors import PosePrior, LandmarkPrior
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    get_measurement_precisions_from_info_matrix,
)
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.data_utils import (
    get_covariance_matrix_from_list,
    load_symmetric_matrix_column_major,
)


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

SE3_VARIABLE = "VERTEX_SE3:QUAT"
SE2_VARIABLE = "VERTEX_SE2:QUAT"
EDGE_SE3 = "EDGE_SE3:QUAT"
EDGE_SE2 = "EDGE_SE2:QUAT"

from attrs import define, field


@define
class Counter:

    count: int = field(default=0)

    def increment(self):
        self.count += 1


C = Counter()


def convert_se3_var_line_to_pose_variable(
    line_items: List[str],
) -> PoseVariable3D:
    """converts the g2o line items for a SE3 variable to a PoseVariable3D object.

    Args:
        line_items (List[str]): List of items in a line of a g2o file.

    Returns:
        PoseVariable3D: PoseVariable3D object corresponding to the line items.
    """
    assert (
        line_items[0] == SE3_VARIABLE
    ), f"Line type is not {SE3_VARIABLE}, it is {line_items[0]}"
    pose_num_idx = 1
    translation_idx_bounds = (2, 5)
    quat_idx_bounds = (5,)
    pose_name = f"A{line_items[pose_num_idx]}"

    # get the translation
    translation_vals = [
        float(x)
        for x in line_items[translation_idx_bounds[0] : translation_idx_bounds[1]]
    ]
    assert len(translation_vals) == 3
    translation = (translation_vals[0], translation_vals[1], translation_vals[2])

    # get the rotation
    quat_vals = [float(x) for x in line_items[quat_idx_bounds[0] :]]
    quat = np.array(quat_vals)
    rot = get_rotation_matrix_from_quat(quat)

    pose_var = PoseVariable3D(pose_name, translation, rot)
    return pose_var


def convert_se3_measurement_line_to_pose_measurement(
    line_items: List[str],
) -> Optional[PoseMeasurement3D]:
    """converts the g2o line items for a SE3 measurement to a PoseMeasurement3D
    object.

    Args:
        line_items (List[str]): List of items in a line of a g2o
        file.

    Returns:
        Optional[PoseMeasurement3D]: PoseMeasurement3D object corresponding to
        the line. Returns None if the measurement is not a movement (no trans or
        rotation)
    """
    assert (
        line_items[0] == EDGE_SE3
    ), f"Line type is not {EDGE_SE3}, it is {line_items[0]}"
    assert len(line_items) == 31, f"Line has {len(line_items)} items, not 31"

    # where the indices are in the line items
    pose_num_idx = 1
    pose_num_idx_2 = 2
    translation_idx_bounds = (3, 6)
    quat_idx_bounds = (6, 10)
    cov_idx_bounds = (10, len(line_items))

    # get the pose names
    from_pose_name = f"A{line_items[pose_num_idx]}"
    to_pose_name = f"A{line_items[pose_num_idx_2]}"

    # get the translation
    translation_vals = [
        float(x)
        for x in line_items[translation_idx_bounds[0] : translation_idx_bounds[1]]
    ]
    translation = np.array(translation_vals)

    # get the rotation
    quat_vals = [float(x) for x in line_items[quat_idx_bounds[0] : quat_idx_bounds[1]]]
    quat = np.array(quat_vals)
    rot = get_rotation_matrix_from_quat(quat)

    no_translation = np.allclose(translation, np.zeros(3))
    no_rotation = np.allclose(rot, np.eye(3))
    if no_translation and no_rotation:
        pass
        # return None

    # parse information matrix
    info_mat_size = 6
    info_vals = [float(x) for x in line_items[cov_idx_bounds[0] : cov_idx_bounds[1]]]
    info_mat = load_symmetric_matrix_column_major(info_vals, info_mat_size)
    trans_precision, rot_precision = get_measurement_precisions_from_info_matrix(
        info_mat, matrix_dim=info_mat_size
    )

    if trans_precision < 0.5 or rot_precision < 0.5:
        err = f"Low precisions! Trans: {trans_precision}, Rot: {rot_precision}"
        C.increment()
        logger.warning(err + f"low-precision factor {C.count}")
        # return None
        # raise ValueError(err)

    # form pose measurement
    pose_measurement = PoseMeasurement3D(
        from_pose_name, to_pose_name, translation, rot, trans_precision, rot_precision
    )
    return pose_measurement


def is_odom_measurement(line_items: List[str]) -> bool:
    """Determine if a line of a g2o file is an odometry measurement.

    Args:
        line_items: List of items in a line of a g2o file.

    Returns:
        True if the line is an odometry measurement, False otherwise.
    """
    assert (
        line_items[0] == EDGE_SE3
    ), f"Line type is not {EDGE_SE3}, it is {line_items[0]}"
    from_idx, to_idx = int(line_items[1]), int(line_items[2])
    return from_idx == to_idx - 1


def parse_3d_g2o_file(filepath: str):
    # read the file line-by-line
    logger.info(f"Parsing 3D g2o file: {filepath}")

    if not isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fg = FactorGraphData(dimension=3)

    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.split()
            line_type = items[0]
            if line_type == SE3_VARIABLE:
                new_pose_var = convert_se3_var_line_to_pose_variable(items)
                fg.add_pose_variable(new_pose_var)
            elif line_type == EDGE_SE3:
                new_pose_measurement = convert_se3_measurement_line_to_pose_measurement(
                    items
                )
                if new_pose_measurement is None:
                    pose_0 = items[1]
                    pose_1 = items[2]
                    logger.warning(
                        f"Skipping measurement between: {pose_0} and {pose_1}"
                    )
                    continue

                if is_odom_measurement(items):
                    robot_idx = 0  # only 1 robot in g2o files
                    fg.add_odom_measurement(robot_idx, new_pose_measurement)
                else:
                    fg.add_loop_closure(new_pose_measurement)
            else:
                raise ValueError(f"Unsupported line type for 3D: {line_type}")

    logger.info(f"Finished parsing 3D g2o file: {filepath}")
    return fg


if __name__ == "__main__":
    from os.path import join, expanduser
    from os import listdir
    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file
    from pathlib import Path

    logger.warning(
        "g2o files do not contain ground truth!! Right now no ground truth implemented"
    )

    np.set_printoptions(precision=3, suppress=True)

    def _get_list_of_g2o_files(dim: int) -> List[str]:
        """Gets a list of all the g2o files in the sesync dataset.

        Returns:
            List[str]: List of paths to the g2o files.
        """
        assert dim in [2, 3], f"Dimension must be 2 or 3, not {dim}"

        base_data_dir = Path(expanduser(f"~/data/g2o/{dim}d"))
        subdirs = [base_data_dir.joinpath(x) for x in listdir(base_data_dir)]
        g2o_files = []
        for subdir in subdirs:
            g2o_files += [
                str(subdir.joinpath(x))
                for x in listdir(subdir)
                if x.endswith(".g2o") and not x.startswith(".")
            ]
        return g2o_files

    g2o_files = _get_list_of_g2o_files(dim=3)
    if len(g2o_files) == 0:
        raise FileNotFoundError("No g2o files found")

    for file in g2o_files:

        pickle_file = file.replace(".g2o", ".pickle")
        if isfile(pickle_file):
            use_choice = ""
            while use_choice not in ["y", "n"]:
                use_choice = input(f"Use existing pickle file? {pickle_file} [y/n]: ")
            if use_choice == "y":
                logger.info(f"Skipping {file}, pickle file already exists")
                continue

        try:
            fg = parse_3d_g2o_file(file)
            fg._save_to_pickle_format(pickle_file)
            pass
        except ValueError as e:
            logger.error(f"Failed parsing file: {file} with error: {e}")
            continue

        fg = parse_pickle_file(pickle_file)
        # fg.print_summary()
