from typing import List, Any, Union, Tuple
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
from py_factor_graph.utils.matrix_utils import get_rotation_matrix_from_quat
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.data_utils import (
    get_covariance_matrix_from_list,
    load_symmetric_matrix_column_major,
)

SE3_VARIABLE = "VERTEX_SE3:QUAT"
SE2_VARIABLE = "VERTEX_SE2:QUAT"
EDGE_SE3 = "EDGE_SE3:QUAT"
EDGE_SE2 = "EDGE_SE2:QUAT"


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
    quat_idx_bounds = (5, 9)
    pose_name = f"A{line_items[pose_num_idx]}"

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

    pose_var = PoseVariable3D(pose_name, translation, rot)
    return pose_var


def convert_se3_measurement_line_to_pose_measurement(
    line_items: List[str],
) -> PoseMeasurement3D:
    """converts the g2o line items for a SE3 measurement to a PoseMeasurement3D
    object.

    Args:
        line_items (List[str]): List of items in a line of a g2o
        file.

    Returns:
        PoseMeasurement3D: PoseMeasurement3D object corresponding to the line
    """
    assert (
        line_items[0] == EDGE_SE3
    ), f"Line type is not {EDGE_SE3}, it is {line_items[0]}"

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

    # parse covariance
    covar_mat_size = 6
    covar_vals = [float(x) for x in line_items[cov_idx_bounds[0] : cov_idx_bounds[1]]]
    covar = load_symmetric_matrix_column_major(covar_vals, covar_mat_size)
    assert (
        covar[0, 0] == covar[1, 1] == covar[2, 2]
        and covar[3, 3] == covar[4, 4] == covar[5, 5]
    ), (
        f"Covariance should be isotropic in translation and rotation"
        f", but it is not. Covariance: {covar}"
    )
    trans_covar = covar[5, 5]
    rot_covar = covar[0, 0]
    trans_weight = 1 / trans_covar
    rot_weight = 1 / rot_covar
    pose_measurement = PoseMeasurement3D(
        from_pose_name, to_pose_name, translation, rot, trans_weight, rot_weight
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
                if is_odom_measurement(items):
                    robot_idx = 0  # only 1 robot in g2o files
                    fg.add_odom_measurement(robot_idx, new_pose_measurement)
                else:
                    fg.add_loop_closure(new_pose_measurement)
            else:
                raise ValueError(f"Unsupported line type for 3D: {line_type}")

    return fg


if __name__ == "__main__":
    file = "/home/alan/data/g2o/grid/grid3D.g2o"
    pickle_file = "/home/alan/data/g2o/grid/grid3D.pkl"
    # fg = parse_3d_g2o_file(file)
    # fg._save_to_pickle_format(pickle_file)

    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file

    fg = parse_pickle_file(pickle_file)

    fg.print_summary()
