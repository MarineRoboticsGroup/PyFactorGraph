from os.path import isfile
import pickle

from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.logging_utils import logger


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
    assert filepath.endswith(".pickle") or filepath.endswith(
        ".pkl"
    ), f"{filepath} is not a pickle file"

    with open(filepath, "rb") as f:
        data = pickle.load(f)
        logger.debug(f"Loaded factor graph data from {filepath}")
        return data
