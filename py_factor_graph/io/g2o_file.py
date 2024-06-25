from typing import List, Optional
from os.path import isfile
import numpy as np

from py_factor_graph.variables import PoseVariable3D, PoseVariable2D
from py_factor_graph.measurements import PoseMeasurement3D, PoseMeasurement2D
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    get_measurement_precisions_from_info_matrix,
    get_symmetric_matrix_from_list_column_major,
)
from py_factor_graph.utils.logging_utils import logger

SE3_VARIABLE = "VERTEX_SE3:QUAT"
SE2_VARIABLE = "VERTEX_SE2"
EDGE_SE3 = "EDGE_SE3:QUAT"
EDGE_SE2 = "EDGE_SE2"

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
    timestep = int(line_items[pose_num_idx])
    pose_name = f"A{timestep}"

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

    pose_var = PoseVariable3D(pose_name, translation, rot, float(timestep))
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

    # timestamp is the same as the greatest pose number
    timestamp = max(int(line_items[pose_num_idx]), int(line_items[pose_num_idx_2]))

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
    info_mat = get_symmetric_matrix_from_list_column_major(info_vals, info_mat_size)
    trans_precision, rot_precision = get_measurement_precisions_from_info_matrix(
        info_mat, matrix_dim=info_mat_size
    )

    if trans_precision < 0.5 or rot_precision < 0.5:
        err = f"Low precisions! Trans: {trans_precision}, Rot: {rot_precision}"
        logger.warning(err + f" low-precision factor {C.count}")

    # form pose measurement
    pose_measurement = PoseMeasurement3D(
        from_pose_name,
        to_pose_name,
        translation,
        rot,
        trans_precision,
        rot_precision,
        float(timestamp),
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
        line_items[0] == EDGE_SE3 or line_items[0] == EDGE_SE2
    ), f"Line type is not {EDGE_SE3} or {EDGE_SE2}, it is {line_items[0]}"
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


def convert_se2_var_line_to_pose_variable(
    line_items: List[str],
) -> PoseVariable2D:
    """converts the g2o line items for a SE2 variable to a PoseVariable3D object.

    Args:
        line_items (List[str]): List of items in a line of a g2o file.

    Returns:
        PoseVariable3D: PoseVariable3D object corresponding to the line items.
    """
    assert (
        line_items[0] == SE2_VARIABLE
    ), f"Line type is not {SE2_VARIABLE}, it is {line_items[0]}"
    pose_num_idx = 1
    translation_idx_bounds = (2, 4)
    theta_idx = 4
    timestamp = int(line_items[pose_num_idx])
    pose_name = f"A{timestamp}"

    # get the translation
    translation_vals = [
        float(x)
        for x in line_items[translation_idx_bounds[0] : translation_idx_bounds[1]]
    ]
    assert len(translation_vals) == 2
    translation = (translation_vals[0], translation_vals[1])

    # get the rotation
    theta = float(line_items[theta_idx])

    pose_var = PoseVariable2D(pose_name, translation, theta, float(timestamp))
    return pose_var


def convert_se2_measurement_line_to_pose_measurement(
    line_items: List[str],
) -> Optional[PoseMeasurement2D]:
    """converts the g2o line items for a SE2 measurement to a PoseMeasurement2D
    object.

    Args:
        line_items (List[str]): List of items in a line of a g2o
        file.

    Returns:
        Optional[PoseMeasurement3D]: PoseMeasurement3D object corresponding to
        the line. Returns None if the measurement is
        not
    """
    assert (
        line_items[0] == EDGE_SE2
    ), f"Line type is not {EDGE_SE2}, it is {line_items[0]}"
    assert len(line_items) == 12, f"Line has {len(line_items)} items, not 12"

    # where the indices are in the line items
    pose_num_idx = 1
    pose_num_idx_2 = 2
    translation_idx_bounds = (3, 5)
    theta_idx = 5
    cov_idx_bounds = (6, len(line_items))

    # get the pose names
    from_pose_name = f"A{line_items[pose_num_idx]}"
    to_pose_name = f"A{line_items[pose_num_idx_2]}"

    # timestamp is the same as the greatest pose number
    timestamp = max(int(line_items[pose_num_idx]), int(line_items[pose_num_idx_2]))

    # get the translation
    translation_vals = [
        float(x)
        for x in line_items[translation_idx_bounds[0] : translation_idx_bounds[1]]
    ]
    translation = np.array(translation_vals)

    # get the rotation
    theta = float(line_items[theta_idx])

    no_translation = np.allclose(translation, np.zeros(2))
    no_rotation = np.allclose(theta, 0)
    if no_translation and no_rotation:
        pass
        # return None

    # parse information matrix
    info_mat_size = 3
    info_vals = [float(x) for x in line_items[cov_idx_bounds[0] : cov_idx_bounds[1]]]
    info_mat = get_symmetric_matrix_from_list_column_major(info_vals, info_mat_size)
    (
        translation_precision,
        theta_precision,
    ) = get_measurement_precisions_from_info_matrix(info_mat, matrix_dim=info_mat_size)

    rpm = PoseMeasurement2D(
        from_pose_name,
        to_pose_name,
        x=translation[0],
        y=translation[1],
        theta=theta,
        translation_precision=translation_precision,
        rotation_precision=theta_precision,
        timestamp=float(timestamp),
    )

    return rpm


def parse_2d_g2o_file(filepath: str):
    # read the file line-by-line
    logger.info(f"Parsing 2D g2o file: {filepath}")

    if not isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fg = FactorGraphData(dimension=2)

    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.split()
            line_type = items[0]
            if line_type == SE2_VARIABLE:
                new_pose_var = convert_se2_var_line_to_pose_variable(items)
                fg.add_pose_variable(new_pose_var)
            elif line_type == EDGE_SE2:
                new_pose_measurement = convert_se2_measurement_line_to_pose_measurement(
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
                raise ValueError(f"Unsupported line type for 2D: {line_type}")

    logger.info(f"Finished parsing 2D g2o file: {filepath}")
    return fg


if __name__ == "__main__":
    from py_factor_graph.modifiers import convert_to_sensor_network_localization

    dirpath = "/home/alan/cora/build/bin/data"
    dirpath = "/home/alan/Downloads"
    # /home/alan/Downloads/grid3D.g2o           /home/alan/Downloads/input_M3500_g2o.g2o  /home/alan/Downloads/sphere_bignoise_vertex3.g2o
    # /home/alan/Downloads/input_INTEL_g2o.g2o  /home/alan/Downloads/input_MITb_g2o.g2o   /home/alan/Downloads/torus3D.g2o
    fname = "sphere_bignoise_vertex3.g2o"
    fname = "torus3D.g2o"
    # fname = "grid3D.g2o"
    fname = "input_M3500_g2o.g2o"
    fname = "input_INTEL_g2o.g2o"
    # fname = "input_MITb_g2o.g2o"

    fpath = f"{dirpath}/{fname}"

    # /home/alan/mrg-mac/data/ais2klinik.g2o
    # /home/alan/mrg-mac/data/intel_edges.g2o
    # /home/alan/mrg-mac/data/kitti_05.g2o
    # /home/alan/mrg-mac/data/sphere2500.g2o
    # /home/alan/mrg-mac/data/city10000.g2o
    # /home/alan/mrg-mac/data/intel.g2o
    # /home/alan/mrg-mac/data/kitti_02.g2o
    # /home/alan/mrg-mac/data/sphere2500_edges.g2o
    dirpath = "/home/alan/mrg-mac/data"
    fname = "ais2klinik.g2o"
    fname = "kitti_05.g2o"
    fname = "sphere2500.g2o"
    # fname = "city10000.g2o"
    # fname = "intel.g2o"
    # fname = "kitti_02.g2o"

    fpath = f"{dirpath}/{fname}"

    files_2d = [
        "input_M3500_g2o.g2o",
        "input_INTEL_g2o.g2o",
        "input_MITb_g2o.g2o",
        "ais2klinik.g2o",
        "intel_edges.g2o",
        "kitti_05.g2o",
        "city10000.g2o",
        "intel.g2o",
        "kitti_02.g2o",
    ]

    if fname in files_2d:
        fg = parse_2d_g2o_file(fpath)
        # fg.animate_odometry(show_gt=True, draw_range_lines=True)
    else:
        fg = parse_3d_g2o_file(fpath)
        # fg.animate_odometry_3d(show_gt=True, draw_range_lines=True)

    new_fg = convert_to_sensor_network_localization(fg)

    from py_factor_graph.io.pyfg_text import save_to_pyfg_text

    new_fg_path = f"/home/alan/cora-plus-plus/build/bin/data/{fname}".replace(
        ".g2o", ".pyfg"
    )
    new_fg_path = "/home/alan/cora-plus-plus/build/bin/data/test.pyfg"
    save_to_pyfg_text(new_fg, new_fg_path)

    # plot all of the landmarks and their ranges
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    num_vars_skip = 10
    for idx, landmark in enumerate(new_fg.landmark_variables):
        if idx % num_vars_skip != 0:
            continue
        continue
        position = landmark.true_position
        ax.scatter(position[0], position[1], position[2], color="r")

    name_to_var_map = new_fg.landmark_variables_dict
    num_measures_skip = 2
    for idx, range_measurement in enumerate(new_fg.range_measurements):
        if idx % num_measures_skip != 0:
            continue
        var1, var2 = range_measurement.association
        loc1 = name_to_var_map[var1].true_position
        loc2 = name_to_var_map[var2].true_position
        xs = [loc1[0], loc2[0]]
        ys = [loc1[1], loc2[1]]
        if len(loc1) == len(loc2) == 3:
            zs = [loc1[2], loc2[2]]  # type: ignore
        else:
            zs = [0.0, 0.0]

        # if len(loc1) == 2 and len(loc2) == 2:
        #     zs = [0.0, 0.0]
        # elif len(loc1) == 3 and len(loc2) == 3:
        #     zs = [loc1[2], loc2[2]]
        # else:
        # raise ValueError("Landmark dimensions not 2 or 3")
        ax.plot(xs, ys, zs, color="b")

    plt.show()
