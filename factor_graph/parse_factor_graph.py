from typing import List
from os.path import isfile
import numpy as np
import pickle

from factor_graph.variables import PoseVariable, LandmarkVariable
from factor_graph.measurements import (
    PoseMeasurement,
    AmbiguousPoseMeasurement,
    FGRangeMeasurement,
    AmbiguousFGRangeMeasurement,
)
from factor_graph.priors import PosePrior, LandmarkPrior
from factor_graph.factor_graph import (
    FactorGraphData,
)
from factor_graph.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from factor_graph.utils.data_utils import get_covariance_matrix_from_list


def parse_efg_file(filepath: str) -> FactorGraphData:
    """
    Parse a factor graph file to extract the factors and variables. Requires
    that the file ends with .fg (e.g. "my_file.fg").

    Args:
        filepath: The path to the factor graph file.

    Returns:
        FactorGraphData: The factor graph data.
    """
    assert isfile(filepath), f"{filepath} is not a file"
    assert filepath.endswith(".fg"), f"{filepath} is not an efg file"

    pose_var_header = "Variable Pose SE2"
    landmark_var_header = "Variable Landmark R2"
    pose_measure_header = "Factor SE2RelativeGaussianLikelihoodFactor"
    amb_measure_header = "Factor AmbiguousDataAssociationFactor"
    range_measure_header = "Factor SE2R2RangeGaussianLikelihoodFactor"
    pose_prior_header = "Factor UnarySE2ApproximateGaussianPriorFactor"
    landmark_prior_header = "Landmark"  # don't have any of these yet

    pose_vars: List[PoseVariable] = []
    landmark_vars: List[LandmarkVariable] = []
    pose_measures: List[List[PoseMeasurement]] = []
    loop_closures: List[PoseMeasurement] = []
    amb_loop_closures: List[AmbiguousPoseMeasurement] = []
    range_measures: List[FGRangeMeasurement] = []
    amb_range_measures: List[AmbiguousFGRangeMeasurement] = []
    pose_priors: List[PosePrior] = []
    landmark_priors: List[LandmarkPrior] = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(pose_var_header):
                line_items = line.split()
                pose_name = line_items[3]
                x = float(line_items[4])
                y = float(line_items[5])
                theta = float(line_items[6])
                pose_vars.append(PoseVariable(pose_name, (x, y), theta))
            elif line.startswith(landmark_var_header):
                line_items = line.split()
                landmark_name = line_items[3]
                x = float(line_items[4])
                y = float(line_items[5])
                landmark_vars.append(LandmarkVariable(landmark_name, (x, y)))
            elif line.startswith(pose_measure_header):
                line_items = line.split()
                base_pose = line_items[2]
                local_pose = line_items[3]
                delta_x = float(line_items[4])
                delta_y = float(line_items[5])
                delta_theta = float(line_items[6])
                covar_list = [float(x) for x in line_items[8:]]
                covar = get_covariance_matrix_from_list(covar_list)
                # assert covar[0, 0] == covar[1, 1]
                trans_weight = 1 / (covar[0, 0])
                rot_weight = 1 / (covar[2, 2])
                measure = PoseMeasurement(
                    base_pose,
                    local_pose,
                    delta_x,
                    delta_y,
                    delta_theta,
                    trans_weight,
                    rot_weight,
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
                    loop_closures.append(measure)

                # otherwise it is an odometry measurement
                else:
                    # make sure that pose_measures has the correct length
                    while base_pose_idx >= len(pose_measures):
                        pose_measures.append([])

                    # add the measurement
                    pose_measures[base_pose_idx].append(
                        PoseMeasurement(
                            base_pose,
                            local_pose,
                            delta_x,
                            delta_y,
                            delta_theta,
                            trans_weight,
                            rot_weight,
                        )
                    )
            elif line.startswith(range_measure_header):
                line_items = line.split()
                var1 = line_items[2]
                var2 = line_items[3]
                dist = float(line_items[4])
                stddev = float(line_items[5])
                range_measures.append(FGRangeMeasurement((var1, var2), dist, stddev))
            elif line.startswith(pose_prior_header):
                line_items = line.split()
                pose_name = line_items[2]
                x = float(line_items[3])
                y = float(line_items[4])
                theta = float(line_items[5])
                covar_list = [float(x) for x in line_items[7:]]
                covar = get_covariance_matrix_from_list(covar_list)
                pose_priors.append(PosePrior(pose_name, (x, y), theta, covar))
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

    return FactorGraphData(
        pose_vars,
        landmark_vars,
        pose_measures,
        loop_closures,
        amb_loop_closures,
        range_measures,
        amb_range_measures,
        pose_priors,
        landmark_priors,
        2,
    )


def parse_pickle_file(filepath: str) -> FactorGraphData:
    """
    Retrieve a pickled FactorGraphData object. Requires that the
    file ends with .pickle (e.g. "my_file.pickle").

    Args:
        filepath: The path to the factor graph file.

    Returns:
        FactorGraphData: The factor graph data.
    """
    assert isfile(filepath), f"{filepath} is not a file"
    assert filepath.endswith(".pickle"), f"{filepath} is not a pickle file"

    with open(filepath, "rb") as f:
        data = pickle.load(f)
        return data
