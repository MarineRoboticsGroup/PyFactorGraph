"""This will write the factor graph to a series of fairly basic graph files.

These files will largely throw away most of the information in the factor graph,
but will retain the structure of the graph. This is useful for:
- visualizing the graph using standard graph visualization tools
- projects that only need the structure of the graph
"""

import numpy as np
import os
import math
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_quat,
    get_symmetric_matrix_from_list_column_major,
    get_list_column_major_from_symmetric_matrix,
    get_measurement_precisions_from_covariance_matrix,
)
from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    LandmarkVariable2D,
    LandmarkVariable3D,
    POSE_VARIABLE_TYPES,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.priors import (
    PosePrior2D,
    PosePrior3D,
    LandmarkPrior2D,
    LandmarkPrior3D,
    POSE_PRIOR_TYPES,
    LANDMARK_PRIOR_TYPES,
)
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    PoseToLandmarkMeasurement2D,
    PoseToLandmarkMeasurement3D,
    FGRangeMeasurement,
    POSE_MEASUREMENT_TYPES,
    POSE_LANDMARK_MEASUREMENT_TYPES,
)
from py_factor_graph.utils.logging_utils import logger

from itertools import chain


def write_edge_list(pyfg: FactorGraphData, output_fpath: str) -> None:
    """Writes the factor graph to a simple edge list file.

    Args:
        pyfg (FactorGraphData): The factor graph to write.
        output_fpath (str): The path to the output file.
    """
    logger.info(f"Writing edge list to {output_fpath}")

    variable_names = pyfg.all_variable_names
    name_to_idx = {name: idx for idx, name in enumerate(variable_names)}

    writer = open(output_fpath, "w")

    # get odometry measurements and flatten them
    odom_measures = chain.from_iterable(pyfg.odom_measurements)
    for odo_measure in odom_measures:
        i, j = name_to_idx[odo_measure.base_pose], name_to_idx[odo_measure.to_pose]
        weight = odo_measure.rotation_precision
        writer.write(f"{i} {j} {weight}\n")

    # loop closures
    loop_closures = pyfg.loop_closure_measurements
    for loop_measure in loop_closures:
        i, j = name_to_idx[loop_measure.base_pose], name_to_idx[loop_measure.to_pose]
        weight = loop_measure.rotation_precision
        writer.write(f"{i} {j} {weight}\n")

    # get pose to landmark measurements
    pose_landmark_measures = pyfg.pose_landmark_measurements
    for pl_measure in pose_landmark_measures:
        i, j = name_to_idx[pl_measure.pose_name], name_to_idx[pl_measure.landmark_name]
        weight = pl_measure.translation_precision
        writer.write(f"{i} {j} {weight}\n")

    # range measurements
    range_measures = pyfg.range_measurements
    for r_measure in range_measures:
        i, j = name_to_idx[r_measure.first_key], name_to_idx[r_measure.second_key]
        weight = r_measure.precision
        writer.write(f"{i} {j} {weight}\n")

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Writes a factor graph to a simple edge list file."
    )
    parser.add_argument(
        "input_graph",
        type=str,
        help="Path to the input .pyfg file containing the factor graph.",
    )
    parser.add_argument(
        "output_edge_list",
        type=str,
        help="Path to the output edge list file (e.g., .edge).",
    )

    args = parser.parse_args()

    from py_factor_graph.io.pyfg_text import read_from_pyfg_text

    # Load the factor graph
    fg = read_from_pyfg_text(args.input_graph)

    # Write the edge list
    write_edge_list(fg, args.output_edge_list)

    print(f"Wrote edge list to {args.output_edge_list}")
