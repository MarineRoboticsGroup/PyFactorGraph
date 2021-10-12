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


def _get_covariance_matrix_from_list(covar_list: List) -> np.ndarray:
    """
    Converts a list of floats to a covariance matrix.

    args:
        covar_list (List): a list of floats representing the covariance matrix

    returns:
        np.ndarray: the covariance matrix
    """
    assert len(covar_list) == 3 * 3, f"{len(covar_list)} != 3x3"
    assert all(isinstance(val, float) for val in covar_list)
    covar_matrix = np.array(
        [
            [covar_list[0], covar_list[1], covar_list[2]],
            [covar_list[3], covar_list[4], covar_list[5]],
            [covar_list[6], covar_list[7], covar_list[8]],
        ]
    )

    assert np.allclose(
        covar_matrix, covar_matrix.T
    ), "Covariance matrix must be symmetric"
    assert covar_matrix.shape == (3, 3), "Covariance matrix must be 3x3"

    return covar_matrix


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
    pose_measures: List[PoseMeasurement] = []
    amb_pose_measures: List[AmbiguousPoseMeasurement] = []
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
                covar = _get_covariance_matrix_from_list(covar_list)
                # assert covar[0, 0] == covar[1, 1]
                trans_weight = 1 / (covar[0, 0])
                rot_weight = 1 / (covar[2, 2])
                pose_measures.append(
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
                covar = _get_covariance_matrix_from_list(covar_list)
                pose_priors.append(PosePrior(pose_name, (x, y), theta, covar))
            elif line.startswith(landmark_prior_header):
                raise NotImplementedError("Landmark priors not implemented yet")
            elif line.startswith(amb_measure_header):
                line_items = line.split()

                # if it is a range measurement then add to ambiguous range
                # measurements list
                if "SE2R2RangeGaussianLikelihoodFactor" in line:
                    raise NotImplementedError("Need to parse for these measurement")

                # if it is a pose measurement then add to ambiguous pose
                # measurements list
                elif "SE2RelativeGaussianLikelihoodFactor" in line:
                    raise NotImplementedError("Need to parse for these measurement")

                # this is a case that we haven't planned for yet
                else:
                    raise NotImplementedError(
                        f"Unknown measurement type in ambiguous measurement: {line}"
                    )

    return FactorGraphData(
        pose_vars,
        landmark_vars,
        pose_measures,
        amb_pose_measures,
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
