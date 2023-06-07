from typing import List, Optional
from os.path import isfile
import numpy as np

from py_factor_graph.variables import PoseVariable3D
from py_factor_graph.measurements import (
    PoseMeasurement3D,
)
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    get_measurement_precisions_from_info_matrix,
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
        logger.warning(err + f" low-precision factor {C.count}")

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
    pass
