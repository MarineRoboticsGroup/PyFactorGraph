from os.path import isfile

from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    FGRangeMeasurement,
    FGBearingMeasurement,
)
from py_factor_graph.priors import (
    PosePrior2D,
)
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.data_utils import get_covariance_matrix_from_list
from py_factor_graph.utils.matrix_utils import (
    get_measurement_precisions_from_covariance_matrix,
)


def parse_efg_file(filepath: str) -> FactorGraphData:
    """
    Parse a factor graph file to extract the factors and variables. Requires
    that the file ends with .fg (e.g. "my_file.fg").

    Args:
        filepath: The path to the factor graph file.

    Returns:
        FactorGraphData: The factor graph data.

    Raises:
        ValueError: If the file does not end with .fg.
        ValueError: If the file does not exist.
    """
    if not isfile(filepath):
        raise ValueError(f"File {filepath} does not exist.")
    if not filepath.endswith(".fg"):
        raise ValueError(f"File {filepath} does not end with .fg.")

    pose_var_header = "Variable Pose SE2"
    landmark_var_header = "Variable Landmark R2"
    pose_measure_header = "Factor SE2RelativeGaussianLikelihoodFactor"
    amb_measure_header = "Factor AmbiguousDataAssociationFactor"
    range_measure_header = "Factor SE2R2RangeGaussianLikelihoodFactor"
    bearing_measure_header = "Factor SE2BearingLikelihoodFactor"
    pose_prior_header = "Factor UnarySE2ApproximateGaussianPriorFactor"
    landmark_prior_header = "Landmark"  # don't have any of these yet

    new_fg_data = FactorGraphData(dimension=2)

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(pose_var_header):
                line_items = line.split()
                pose_name = line_items[3]
                x = float(line_items[4])
                y = float(line_items[5])
                theta = float(line_items[6])
                pose_var = PoseVariable2D(pose_name, (x, y), theta)
                new_fg_data.add_pose_variable(pose_var)
            elif line.startswith(landmark_var_header):
                line_items = line.split()
                landmark_name = line_items[3]
                x = float(line_items[4])
                y = float(line_items[5])
                landmark_var = LandmarkVariable2D(landmark_name, (x, y))
                new_fg_data.add_landmark_variable(landmark_var)
            elif line.startswith(pose_measure_header):
                line_items = line.split()
                base_pose = line_items[2]
                local_pose = line_items[3]
                delta_x = float(line_items[4])
                delta_y = float(line_items[5])
                delta_theta = float(line_items[6])
                covar_list = [float(x) for x in line_items[8:]]
                covar = get_covariance_matrix_from_list(covar_list)
                (
                    trans_precision,
                    rot_precision,
                ) = get_measurement_precisions_from_covariance_matrix(
                    covar, matrix_dim=3
                )
                measure = PoseMeasurement2D(
                    base_pose,
                    local_pose,
                    delta_x,
                    delta_y,
                    delta_theta,
                    trans_precision,
                    rot_precision,
                )

                base_pose_idx = get_robot_idx_from_frame_name(base_pose)
                local_pose_idx = get_robot_idx_from_frame_name(local_pose)
                base_time_idx = get_time_idx_from_frame_name(base_pose)
                local_time_idx = get_time_idx_from_frame_name(local_pose)

                # if either the robot indices are different or the time indices
                # are not sequential then it is a loop closure
                if (
                    base_pose_idx != local_pose_idx
                    or local_time_idx != base_time_idx + 1
                ):
                    new_fg_data.add_loop_closure(measure)

                # otherwise it is an odometry measurement
                else:
                    new_fg_data.add_odom_measurement(base_pose_idx, measure)

            elif line.startswith(range_measure_header):
                line_items = line.split()
                var1 = line_items[2]
                var2 = line_items[3]
                dist = float(line_items[4])
                stddev = float(line_items[5])
                range_measure = FGRangeMeasurement((var1, var2), dist, stddev)
                new_fg_data.add_range_measurement(range_measure)

            elif line.startswith(bearing_measure_header):
                line_items = line.split()
                var1 = line_items[2]
                var2 = line_items[3]
                bearing_azimuth = float(line_items[4])
                bearing_elevation = float(line_items[5])
                azimuth_stddev = float(line_items[6])
                elevation_stddev = float(line_items[7])
                bearing_measure = FGBearingMeasurement((var1, var2), bearing_azimuth, bearing_elevation, azimuth_stddev, elevation_stddev)
                new_fg_data.add_bearing_measurement(bearing_measure)

            elif line.startswith(pose_prior_header):
                line_items = line.split()
                pose_name = line_items[2]
                x = float(line_items[3])
                y = float(line_items[4])
                theta = float(line_items[5])
                covar_list = [float(x) for x in line_items[7:]]
                covar = get_covariance_matrix_from_list(covar_list)
                (
                    translation_precision,
                    rotation_precision,
                ) = get_measurement_precisions_from_covariance_matrix(
                    covar, matrix_dim=3
                )
                pose_prior = PosePrior2D(
                    pose_name, (x, y), theta, translation_precision, rotation_precision
                )
                new_fg_data.add_pose_prior(pose_prior)

            elif line.startswith(landmark_prior_header):
                raise NotImplementedError("Landmark priors not implemented yet")
            elif line.startswith(amb_measure_header):
                line_items = line.split()

                # if it is a range measurement then add to ambiguous range
                # measurements list
                if "SE2R2RangeGaussianLikelihoodFactor" in line:
                    raise NotImplementedError(
                        "Need to parse for ambiguous range measurements measurement"
                    )

                # if it is a pose measurement then add to ambiguous pose
                # measurements list
                elif "SE2RelativeGaussianLikelihoodFactor" in line:
                    raise NotImplementedError(
                        "Need to parse for ambiguous pose measurement"
                    )

                # this is a case that we haven't planned for yet
                else:
                    raise NotImplementedError(
                        f"Unknown measurement type in ambiguous measurement: {line}"
                    )

    return new_fg_data
