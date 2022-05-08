from typing import List
from os.path import isfile
import numpy as np
import pickle

from py_factor_graph.variables import PoseVariable, LandmarkVariable
from py_factor_graph.measurements import (
    PoseMeasurement,
    AmbiguousPoseMeasurement,
    FGRangeMeasurement,
    AmbiguousFGRangeMeasurement,
)
from py_factor_graph.priors import PosePrior, LandmarkPrior
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.data_utils import get_covariance_matrix_from_list


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
