import numpy as np
import math
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    load_symmetric_matrix_column_major,
    convert_symmetric_matrix_to_list_column_major,
    get_measurement_precisions_from_covariance_matrix,
)
from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    POSE_VARIABLE_TYPES,
    LandmarkVariable2D,
    LandmarkVariable3D,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    PoseToLandmarkMeasurement2D,
    PoseToLandmarkMeasurement3D,
    POSE_MEASUREMENT_TYPES,
    POSE_TO_LANDMARK_MEASUREMENT_TYPES,
    FGRangeMeasurement,
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


POSE_TYPE_2D = "VERTEX_SE2"
POSE_TYPE_3D = "VERTEX_SE3:QUAT"
LANDMARK_TYPE_2D = "VERTEX_XY"
LANDMARK_TYPE_3D = "VERTEX_XYZ"
REL_POSE_POSE_TYPE_2D = "EDGE_SE2"
REL_POSE_POSE_TYPE_3D = "EDGE_SE3:QUAT"
REL_POSE_LANDMARK_TYPE_2D = "EDGE_SE2_XY"
REL_POSE_LANDMARK_TYPE_3D = "EDGE_SE3_XYZ"
RANGE_MEASURE_TYPE = "EDGE_RANGE"


def _num_elems_symmetric_matrix(dim: int) -> int:
    """Get the number of elements in a symmetric matrix of dimension dim.

    Args:
        dim (int): dimension of the matrix.

    Returns:
        int: number of elements in the matrix.
    """
    return int(dim * (dim + 1) / 2)


def save_to_pyfg_text(fg: FactorGraphData, fpath: str):
    """Save factor graph to g2o file format.

    Args:
        fg (FactorGraphData): factor graph data.
        fpath (str): file path.
    """
    if fg.dimension == 2:
        pose_type = POSE_TYPE_2D
        landmark_type = LANDMARK_TYPE_2D
        rel_pose_pose_type = REL_POSE_POSE_TYPE_2D
        rel_pose_landmark_type = REL_POSE_LANDMARK_TYPE_2D
    elif fg.dimension == 3:
        pose_type = POSE_TYPE_3D
        landmark_type = LANDMARK_TYPE_3D
        rel_pose_pose_type = REL_POSE_POSE_TYPE_3D
        rel_pose_landmark_type = REL_POSE_LANDMARK_TYPE_3D

    range_measure_type = RANGE_MEASURE_TYPE

    def _get_pose_var_string(pose_var: POSE_VARIABLE_TYPES):
        if isinstance(pose_var, PoseVariable2D):
            x, y = pose_var.true_position
            theta = pose_var.true_theta
            return f"{pose_type} {pose_var.name} {x} {y} {theta}"
        elif isinstance(pose_var, PoseVariable3D):
            x, y, z = pose_var.true_position
            qx, qy, qz, qw = pose_var.true_quat
            return f"{pose_type} {pose_var.name} {x} {y} {z} {qx} {qy} {qz} {qw}"
        else:
            raise ValueError(f"Unknown pose type {type(pose_var)}")

    def _get_landmark_var_string(landmark_var: LANDMARK_VARIABLE_TYPES):
        if isinstance(landmark_var, LandmarkVariable2D):
            x, y = landmark_var.true_position
            return f"{landmark_type} {landmark_var.name} {x} {y}"
        elif isinstance(landmark_var, LandmarkVariable3D):
            x, y, z = landmark_var.true_position
            return f"{landmark_type} {landmark_var.name} {x} {y} {z}"
        else:
            raise ValueError(f"Unknown landmark type {type(landmark_var)}")

    def _get_pose_pose_measure_string(measure: POSE_MEASUREMENT_TYPES):
        measurement_connectivity = f"{measure.base_pose} {measure.to_pose}"
        covar_mat = measure.covariance
        covar_mat_elems = convert_symmetric_matrix_to_list_column_major(covar_mat)
        pose_pose_measure_dim = 3 if isinstance(measure, PoseMeasurement2D) else 6
        assert len(covar_mat_elems) == _num_elems_symmetric_matrix(
            pose_pose_measure_dim
        )
        if isinstance(measure, PoseMeasurement2D):
            measurement_values = f"{measure.x} {measure.y} {measure.theta}"
            measurement_noise = " ".join([str(x) for x in covar_mat_elems])
            return f"{rel_pose_pose_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
        elif isinstance(measure, PoseMeasurement3D):
            quat = measure.quat
            qx, qy, qz, qw = quat
            measurement_values = (
                f"{measure.x} {measure.y} {measure.z} {qx} {qy} {qz} {qw}"
            )
            measurement_noise = " ".join([str(x) for x in covar_mat_elems])
            return f"{rel_pose_pose_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
        else:
            raise ValueError(f"Unknown measurement type {type(measure)}")

    def _get_pose_landmark_measure_string(measure: POSE_TO_LANDMARK_MEASUREMENT_TYPES):
        measurement_connectivity = f"{measure.pose_name} {measure.landmark_name}"
        cover_mat = measure.covariance
        covar_mat_elems = convert_symmetric_matrix_to_list_column_major(cover_mat)
        pose_landmark_measure_dim = (
            2 if isinstance(measure, PoseToLandmarkMeasurement2D) else 3
        )
        assert len(covar_mat_elems) == _num_elems_symmetric_matrix(
            pose_landmark_measure_dim
        )
        if isinstance(measure, PoseToLandmarkMeasurement2D):
            measurement_values = f"{measure.x} {measure.y}"
            measurement_noise = " ".join([str(x) for x in covar_mat_elems])
            full_string = f"{rel_pose_landmark_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
            return full_string
        elif isinstance(measure, PoseToLandmarkMeasurement3D):
            measurement_values = f"{measure.x} {measure.y} {measure.z}"
            measurement_noise = " ".join([str(x) for x in covar_mat_elems])
            full_string = f"{rel_pose_landmark_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
            return full_string
        else:
            raise ValueError(f"Unknown measurement type {type(measure)}")

    with open(fpath, "w") as f:
        for pose_chain in fg.pose_variables:
            for pose in pose_chain:
                f.write(_get_pose_var_string(pose) + "\n")

        for landmark in fg.landmark_variables:
            f.write(_get_landmark_var_string(landmark) + "\n")

        for odom_chain in fg.odom_measurements:
            for odom in odom_chain:
                f.write(_get_pose_pose_measure_string(odom) + "\n")

        for loop_closure in fg.loop_closure_measurements:
            if isinstance(loop_closure, (PoseMeasurement2D, PoseMeasurement3D)):
                f.write(_get_pose_pose_measure_string(loop_closure) + "\n")
            elif isinstance(
                loop_closure, (PoseToLandmarkMeasurement2D, PoseToLandmarkMeasurement3D)
            ):
                f.write(_get_pose_landmark_measure_string(loop_closure) + "\n")
            else:
                raise ValueError(f"Unknown measurement type {type(loop_closure)}")

        for range_measure in fg.range_measurements:
            f.write(
                f"{range_measure_type} {range_measure.pose_key} {range_measure.landmark_key} {range_measure.dist} {range_measure.variance}\n"
            )

    logger.info(f"Saved factor graph in PyFG text format to {fpath}")


def read_from_pyfg_text(fpath: str) -> FactorGraphData:
    # assume that dimension is consistent

    # quickly read the first line to get the dimension
    f_temp = open(fpath, "r")

    def _get_dim_from_first_line(line: str) -> int:
        if line.startswith(POSE_TYPE_2D):
            return 2
        elif line.startswith(POSE_TYPE_3D):
            return 3
        else:
            raise ValueError(f"Unknown pose type {line}")

    dim = _get_dim_from_first_line(f_temp.readline())
    f_temp.close()

    # make different types an enum: {POSE, LANDMARK, REL_POSE_POSE, REL_POSE_LANDMARK, RANGE_MEASURE}
    line_types = {
        POSE_TYPE_2D: "POSE",
        POSE_TYPE_3D: "POSE",
        LANDMARK_TYPE_2D: "LANDMARK",
        LANDMARK_TYPE_3D: "LANDMARK",
        REL_POSE_POSE_TYPE_2D: "REL_POSE_POSE",
        REL_POSE_POSE_TYPE_3D: "REL_POSE_POSE",
        REL_POSE_LANDMARK_TYPE_2D: "REL_POSE_LANDMARK",
        REL_POSE_LANDMARK_TYPE_3D: "REL_POSE_LANDMARK",
        RANGE_MEASURE_TYPE: "RANGE_MEASURE",
    }

    def _get_pose_var_from_line(line: str) -> POSE_VARIABLE_TYPES:
        line_parts = line.split(" ")
        pose_var_name = line_parts[1]
        if dim == 2:
            # f"{pose_type} {pose_var.name} {x} {y} {theta}"
            assert line_parts[0] == POSE_TYPE_2D
            assert len(line_parts) == 5
            x, y, theta = [float(x) for x in line_parts[2:]]
            return PoseVariable2D(
                name=pose_var_name, true_position=(x, y), true_theta=theta
            )

        elif dim == 3:
            # f"{pose_type} {pose_var.name} {x} {y} {z} {qx} {qy} {qz} {qw}"
            assert line_parts[0] == POSE_TYPE_3D
            assert len(line_parts) == 9
            x, y, z, qx, qy, qz, qw = [float(x) for x in line_parts[2:]]
            quat = np.array([qx, qy, qz, qw])
            rot_mat = get_rotation_matrix_from_quat(quat)
            return PoseVariable3D(
                name=pose_var_name,
                true_position=(x, y, z),
                true_rotation=rot_mat,
                timestamp=None,
            )
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _get_landmark_var_from_line(line: str) -> LANDMARK_VARIABLE_TYPES:
        line_parts = line.split(" ")
        landmark_var_name = line_parts[1]
        if dim == 2:
            # f"{landmark_type} {landmark_var.name} {x} {y}"
            assert line_parts[0] == LANDMARK_TYPE_2D
            assert len(line_parts) == 4
            x, y = [float(x) for x in line_parts[2:]]
            return LandmarkVariable2D(name=landmark_var_name, true_position=(x, y))
        elif dim == 3:
            # f"{landmark_type} {landmark_var.name} {x} {y} {z}"
            assert line_parts[0] == LANDMARK_TYPE_3D
            assert len(line_parts) == 5
            x, y, z = [float(x) for x in line_parts[2:]]
            return LandmarkVariable3D(name=landmark_var_name, true_position=(x, y, z))
        else:
            raise ValueError(f"Unknown dimension {dim}")

    pose_pose_measure_dim = 3 if dim == 2 else 6
    num_trans_and_rot_entries = (
        3 if dim == 2 else 7
    )  # 3 for 2D, 7 for 3D (quaternion representation)
    pose_pose_measure_noise_dim = _num_elems_symmetric_matrix(pose_pose_measure_dim)

    def _get_pose_pose_measure_from_line(line: str) -> POSE_MEASUREMENT_TYPES:
        line_parts = line.split(" ")
        base_pose_name = line_parts[1]
        to_pose_name = line_parts[2]
        measurement = [float(x) for x in line_parts[3 : 3 + num_trans_and_rot_entries]]
        covar_elements = [float(x) for x in line_parts[3 + num_trans_and_rot_entries :]]
        covar_mat = load_symmetric_matrix_column_major(
            covar_elements, pose_pose_measure_dim
        )
        (
            trans_precision,
            rot_precision,
        ) = get_measurement_precisions_from_covariance_matrix(
            covar_mat, matrix_dim=pose_pose_measure_dim
        )
        assert (
            len(line_parts) == 3 + pose_pose_measure_dim + pose_pose_measure_noise_dim
        )
        if dim == 2:
            assert line_parts[0] == REL_POSE_POSE_TYPE_2D
            # full_string = f"{rel_pose_landmark_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
            x, y, theta = measurement
            return PoseMeasurement2D(
                base_pose=base_pose_name,
                to_pose=to_pose_name,
                x=x,
                y=y,
                theta=theta,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
            )
        elif dim == 3:
            # full_string = f"{rel_pose_landmark_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
            assert line_parts[0] == REL_POSE_POSE_TYPE_3D
            x, y, z, qx, qy, qz, qw = [float(x) for x in line_parts[3:3]]
            translation = np.array([x, y, z])
            quat = np.array([qx, qy, qz, qw])
            rot_mat = get_rotation_matrix_from_quat(quat)
            covar_elements = [float(x) for x in line_parts[10:]]
            covar_mat = load_symmetric_matrix_column_major(covar_elements, 6)
            (
                trans_precision,
                rot_precision,
            ) = get_measurement_precisions_from_covariance_matrix(
                covar_mat, matrix_dim=6
            )
            return PoseMeasurement3D(
                base_pose=base_pose_name,
                to_pose=to_pose_name,
                translation=translation,
                rotation=rot_mat,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
            )
        else:
            raise ValueError(f"Unknown dimension {dim}")

    pose_landmark_measure_dim = 2 if dim == 2 else 3
    pose_landmark_measure_noise_dim = _num_elems_symmetric_matrix(
        pose_landmark_measure_dim
    )

    def _get_pose_landmark_measure_from_line(
        line: str,
    ) -> POSE_TO_LANDMARK_MEASUREMENT_TYPES:
        line_parts = line.split(" ")
        assert (
            len(line_parts)
            == 3 + pose_landmark_measure_dim + pose_landmark_measure_noise_dim
        )
        pose_name = line_parts[1]
        landmark_name = line_parts[2]
        measurement = [float(x) for x in line_parts[3 : 3 + pose_landmark_measure_dim]]
        covar_elements = [float(x) for x in line_parts[3 + pose_landmark_measure_dim :]]
        assert len(covar_elements) == pose_landmark_measure_noise_dim
        covar_mat = load_symmetric_matrix_column_major(
            covar_elements, pose_landmark_measure_dim
        )
        # augment the covariance matrix with identity matrix on block diagonal
        # to be same dim as pose_pose_measure
        augmented_covar_mat = np.eye(pose_pose_measure_dim)
        augmented_covar_mat[
            :pose_landmark_measure_dim, :pose_landmark_measure_dim
        ] = covar_mat
        (
            trans_precision,
            rot_precision,
        ) = get_measurement_precisions_from_covariance_matrix(
            augmented_covar_mat, matrix_dim=pose_pose_measure_dim
        )
        if dim == 2:
            assert line_parts[0] == REL_POSE_LANDMARK_TYPE_2D
            # full_string = f"{rel_pose_landmark_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
            x, y = measurement
            return PoseToLandmarkMeasurement2D(
                pose_name=pose_name,
                landmark_name=landmark_name,
                x=x,
                y=y,
                translation_precision=trans_precision,
            )
        elif dim == 3:
            assert line_parts[0] == REL_POSE_LANDMARK_TYPE_3D
            # full_string = f"{rel_pose_landmark_type} {measurement_connectivity} {measurement_values} {measurement_noise}"
            x, y, z = measurement
            return PoseToLandmarkMeasurement3D(
                pose_name=pose_name,
                landmark_name=landmark_name,
                x=x,
                y=y,
                z=z,
                translation_precision=trans_precision,
            )
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _get_range_measure_from_line(line: str) -> FGRangeMeasurement:
        # f"{range_measure_type} {range_measure.pose_key} {range_measure.landmark_key} {range_measure.dist} {range_measure.variance}\n"
        line_parts = line.split(" ")
        assert len(line_parts) == 5
        assert line_parts[0] == RANGE_MEASURE_TYPE
        association = (line_parts[1], line_parts[2])
        dist = float(line_parts[3])
        variance = float(line_parts[4])
        return FGRangeMeasurement(
            association=association, dist=dist, stddev=math.sqrt(variance)
        )

    def _rel_pose_pose_is_odom(measure: POSE_MEASUREMENT_TYPES) -> bool:
        base_pose_robot_idx = get_robot_idx_from_frame_name(measure.base_pose)
        to_pose_robot_idx = get_robot_idx_from_frame_name(measure.to_pose)
        if not base_pose_robot_idx == to_pose_robot_idx:
            return False

        base_pose_time_idx = get_time_idx_from_frame_name(measure.base_pose)
        to_pose_time_idx = get_time_idx_from_frame_name(measure.to_pose)
        if not base_pose_time_idx + 1 == to_pose_time_idx:
            return False

        return True

    pyfg = None
    f = open(fpath, "r")

    # iterate over lines
    for line in f:
        if pyfg is None:
            dim = _get_dim_from_first_line(line)
            pyfg = FactorGraphData(dimension=dim)

        line = line.strip()
        line_type = line_types[line.split(" ")[0]]

        if line_type == "POSE":
            pose_var = _get_pose_var_from_line(line)
            pyfg.add_pose_variable(pose_var)
        elif line_type == "LANDMARK":
            landmark_var = _get_landmark_var_from_line(line)
            pyfg.add_landmark_variable(landmark_var)
        elif line_type == "REL_POSE_POSE":
            pose_measure = _get_pose_pose_measure_from_line(line)
            # check if is odom (i.e., the first char is the same and the remainder of the names has a difference of 1)
            if _rel_pose_pose_is_odom(pose_measure):
                robot_idx = get_robot_idx_from_frame_name(pose_measure.base_pose)
                pyfg.add_odom_measurement(robot_idx, pose_measure)
            else:
                pyfg.add_loop_closure(pose_measure)
        elif line_type == "REL_POSE_LANDMARK":
            raise NotImplementedError(
                "We don't support relative pose to landmark measurements yet"
            )
            pose_landmark_measure = _get_pose_landmark_measure_from_line(line)
            pyfg.add_loop_closure(pose_landmark_measure)
        elif line_type == "RANGE_MEASURE":
            range_measure = _get_range_measure_from_line(line)
            pyfg.add_range_measurement(range_measure)

    f.close()

    assert isinstance(pyfg, FactorGraphData)
    logger.info(f"Loaded factor graph in PyFG text format from {fpath}")
    return pyfg


if __name__ == "__main__":
    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file
    from os.path import expanduser

    sample_file = expanduser("~/experimental_data/plaza/Plaza1/factor_graph.pickle")

    fg = parse_pickle_file(sample_file)
    fg.print_summary()
    save_to_pyfg_text(fg, "/tmp/test.txt")
    fg.animate_odometry(
        show_gt=True,
        draw_range_lines=True,
        draw_range_circles=True,
        num_timesteps_keep_ranges=10,
    )

    fg2 = read_from_pyfg_text("/tmp/test.txt")
    fg2.print_summary()
    fg2.animate_odometry(
        show_gt=True,
        draw_range_lines=True,
        draw_range_circles=True,
        num_timesteps_keep_ranges=10,
    )
