from typing import List, Dict, Set, Optional, Tuple, Union
import attr
import pickle
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import mpl_toolkits.mplot3d.art3d as art3d

from py_factor_graph.utils.matrix_utils import (
    get_translation_from_transformation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_quat_from_rotation_matrix,
)

from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    LandmarkVariable2D,
    LandmarkVariable3D,
    POSE_VARIABLE_TYPES,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    AmbiguousPoseMeasurement2D,
    PoseToLandmarkMeasurement2D,
    PoseToLandmarkMeasurement3D,
    FGRangeMeasurement,
    AmbiguousFGRangeMeasurement,
    FGBearingMeasurement,
    POSE_MEASUREMENT_TYPES,
    POSE_LANDMARK_MEASUREMENT_TYPES,
)
from py_factor_graph.priors import (
    PosePrior2D,
    PosePrior3D,
    LandmarkPrior2D,
    LandmarkPrior3D,
    POSE_PRIOR_TYPES,
    LANDMARK_PRIOR_TYPES,
)
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.plot_utils import (
    draw_pose,
    draw_pose_3d,
    update_pose_arrow,
    draw_traj,
    draw_traj_3d,
    draw_landmark_variable,
    draw_landmark_variable_3d,
    draw_range_measurement,
    draw_range_measurement_3d,
)
from py_factor_graph.utils.attrib_utils import is_dimension
from py_factor_graph.utils.logging_utils import logger


@attr.s
class FactorGraphData:
    """
    Just a container for the data in a FactorGraph. Only considers standard
    gaussian measurements.

    Ambiguous measurements are used to represent cases where data association
    was uncertain.

    Args:
        pose_variables (List[List[POSE_VARIABLE_TYPES]]): the pose chains. Each
        different robot is a different one of the nested lists.
        landmark_variables (List[LANDMARK_VARIABLE_TYPES]): the landmark variables
        odom_measurements (List[List[PoseMeasurement2D]]): the odom measurements.
        Same structure as pose_variables.
        loop_closure_measurements (List[PoseMeasurement2D]): the loop closures.
        ambiguous_loop_closure_measurements (List[AmbiguousPoseMeasurement2D]): the ambiguous loop closures.
        pose_landmark_measurements (List[POSE_LANDMARK_MEASUREMENT_TYPES]): the pose to landmark measurements.
        range_measurements (List[FGRangeMeasurement]): the range measurements.
        bearing_measurements (List[FGBearingMeasurement]): the bearing measurements.
        ambiguous_range_measurements (List[AmbiguousFGRangeMeasurement]): the ambiguous range measurements.
        pose_priors (List[PosePrior2D]): the pose priors.
        landmark_priors (List[LandmarkPrior2D]): the landmark priors.
        dimension (int): the dimension of the factor graph (e.g. 3 for 3D).

    Raises:
        ValueError: inputs do not match criteria.
    """

    # latent dimension of the space (e.g. 2D or 3D)
    dimension: int = attr.ib(validator=is_dimension)

    # variables
    pose_variables: List[List[POSE_VARIABLE_TYPES]] = attr.ib(factory=list)
    landmark_variables: List[LANDMARK_VARIABLE_TYPES] = attr.ib(factory=list)
    existing_pose_variables: Set[str] = attr.ib(factory=set)
    existing_landmark_variables: Set[str] = attr.ib(factory=set)

    # pose measurements
    odom_measurements: List[List[POSE_MEASUREMENT_TYPES]] = attr.ib(factory=list)
    loop_closure_measurements: List[POSE_MEASUREMENT_TYPES] = attr.ib(factory=list)
    ambiguous_loop_closure_measurements: List[AmbiguousPoseMeasurement2D] = attr.ib(
        factory=list
    )  # TODO: extend to AmbiguousPoseMeasurement3D

    # pose to landmark measurements
    pose_landmark_measurements: List[POSE_LANDMARK_MEASUREMENT_TYPES] = attr.ib(
        factory=list
    )
    # TODO: add ambiguous pose_landmark_measurements

    # range measurements
    range_measurements: List[FGRangeMeasurement] = attr.ib(factory=list)
    ambiguous_range_measurements: List[AmbiguousFGRangeMeasurement] = attr.ib(
        factory=list
    )

    # bearing measurements
    bearing_measurements: List[FGBearingMeasurement] = attr.ib(factory=list)

    # priors
    pose_priors: List[POSE_PRIOR_TYPES] = attr.ib(factory=list)
    landmark_priors: List[LANDMARK_PRIOR_TYPES] = attr.ib(factory=list)

    # useful helper values
    x_min: Optional[float] = attr.ib(default=None)
    x_max: Optional[float] = attr.ib(default=None)
    y_min: Optional[float] = attr.ib(default=None)
    y_max: Optional[float] = attr.ib(default=None)
    z_min: Optional[float] = attr.ib(default=None)
    z_max: Optional[float] = attr.ib(default=None)
    max_measure_weight: Optional[float] = attr.ib(default=None)
    min_measure_weight: Optional[float] = attr.ib(default=None)

    def __str__(self):
        line = "Factor Graph Data\n"

        # add pose variables
        line += f"Pose Variables: {len(self.pose_variables)}\n"
        for x in self.pose_variables:
            line += f"{x}\n"
        line += "\n"

        # add landmarks
        line += f"Landmark Variables: {len(self.landmark_variables)}\n"
        for x in self.landmark_variables:
            line += f"{x}\n"
        line += "\n"

        # add odom measurements
        line += f"Odom Measurements: {len(self.odom_measurements)}\n"
        for x in self.odom_measurements:
            line += f"{x}\n"
        line += "\n"

        # add loop closure measurements
        line += f"Loop Closure Measurements: {len(self.loop_closure_measurements)}\n"
        for x in self.loop_closure_measurements:
            line += f"{x}\n"
        line += "\n"

        # add ambiguous loop closure measurements
        line += f"Ambiguous Loop Closure Measurements: {len(self.ambiguous_loop_closure_measurements)}\n"
        for x in self.ambiguous_loop_closure_measurements:
            line += f"{x}\n"
        line += "\n"

        # add pose to landmark measurements
        line += (
            f"Pose to landmark measurements: {len(self.pose_landmark_measurements)}\n"
        )
        for x in self.pose_landmark_measurements:
            line += f"{x}\n"
        line += "\n"

        # add range measurements
        line += f"Range Measurements: {len(self.range_measurements)}\n"
        for x in self.range_measurements:
            line += f"{x}\n"
        line += "\n"

        # add ambiguous range measurements
        line += (
            f"Ambiguous Range Measurements: {len(self.ambiguous_range_measurements)}\n"
        )
        for x in self.ambiguous_range_measurements:
            line += f"{x}\n"
        line += "\n"

        # add bearing measurements
        line += f"Bearing Measurements: {len(self.bearing_measurements)}\n"
        for x in self.bearing_measurements:
            line += f"{x}\n"
        line += "\n"

        # add pose priors
        line += f"Pose Priors: {len(self.pose_priors)}\n"
        for x in self.pose_priors:
            line += f"{x}\n"
        line += "\n"

        # add landmark priors
        line += f"Landmark Priors: {len(self.landmark_priors)}\n"
        for x in self.landmark_priors:
            line += f"{x}\n"
        line += "\n"

        # add dimension
        line += f"Dimension: {self.dimension}\n\n"
        return line

    def num_poses_by_robot_idx(self, robot_idx: int) -> int:
        """Returns the number of pose variables for a given robot index.

        Args:
            robot_idx (int): the robot index

        Returns:
            int: the number of pose variables for the given robot index
        """

        # if there are no pose variables, return 0
        if len(self.pose_variables) <= robot_idx:
            return 0

        return len(self.pose_variables[robot_idx])

    def print_summary(self) -> str:
        """Prints a summary of the factor graph data."""
        num_robots = self.num_robots
        num_poses = self.num_poses
        num_landmarks = self.num_landmarks
        num_odom_measurements = self.num_odom_measurements
        num_pose_landmark_measurements = self.num_pose_landmark_measurements
        num_range_measurements = self.num_range_measurements
        num_bearing_measurements = self.num_bearing_measurements
        num_loop_closures = self.num_loop_closures
        robots_line = f"Robots: {num_robots}"
        variables_line = f"Variables: {num_poses} poses, {num_landmarks} landmarks"
        measurements_line = (
            f"Measurements: {num_odom_measurements} odom, "
            f"{num_pose_landmark_measurements} pose to landmark, "
            f"{num_bearing_measurements} bearing, "
            f"{num_range_measurements} range, {num_loop_closures} loop closures "
            f"Interrobot loop closures: {self.interrobot_loop_closure_info}"
        )
        msg = f"{robots_line} || {variables_line} || {measurements_line}"
        logger.info(msg)
        return msg

    @property
    def num_robots(self) -> int:
        """Returns the number of robots.

        Returns:
            int: the number of robots
        """
        non_zero_odom_chains = [x for x in self.odom_measurements if len(x) > 0]
        return len(non_zero_odom_chains)

    @property
    def is_empty(self) -> bool:
        """Returns whether the factor graph data is empty.

        Returns:
            bool: whether the factor graph data is empty
        """
        # if there are no pose variables, return True
        return self.num_poses == 0

    @property
    def num_poses(self) -> int:
        """Returns the number of pose variables.

        Returns:
            int: the number of pose variables
        """
        if len(self.pose_variables) == 0:
            return 0

        return sum([len(x) for x in self.pose_variables])

    @property
    def num_landmarks(self) -> int:
        """Returns the number of landmark variables.

        Returns:
            int: the number of landmark variables
        """
        return len(self.landmark_variables)

    @property
    def num_odom_measurements(self) -> int:
        """Returns the number of odometry measurements.

        Returns:
            int: the number of odometry measurements
        """
        return sum([len(x) for x in self.odom_measurements])

    @property
    def num_loop_closures(self) -> int:
        """Returns the number of loop closure measurements.

        Returns:
            int: the number of loop closure measurements
        """
        return len(self.loop_closure_measurements)

    @property
    def odom_precisions(self) -> List[Tuple[float, float]]:
        """Returns the odometry precisions.

        Returns:
            List[Tuple[float, float]]: the odometry precisions
        """
        precisions = []
        for odom_chain in self.odom_measurements:
            for odom in odom_chain:
                precisions.append((odom.translation_precision, odom.rotation_precision))
        return precisions

    @property
    def num_interrobot_loop_closures(self) -> int:
        """Returns the number of inter-robot loop closure measurements.

        Returns:
            int: the number of inter-robot loop closure measurements
        """
        return len(self.interrobot_loop_closures)

    @property
    def interrobot_loop_closure_info(self) -> str:
        """Returns a string containing information about the inter-robot loop closures.

        Returns:
            str: a string containing information about the inter-robot loop closures
        """
        if len(self.interrobot_loop_closures) == 0:
            return "No inter-robot loop closures"

        loop_closure_counts: Dict[Tuple[str, str], int] = {}
        for closure in self.interrobot_loop_closures:
            base_pose_char = closure.base_pose[0]
            to_pose_char = closure.to_pose[0]
            if base_pose_char > to_pose_char:
                base_pose_char, to_pose_char = to_pose_char, base_pose_char
            association = (base_pose_char, to_pose_char)
            loop_closure_counts[association] = (
                loop_closure_counts.get(association, 0) + 1
            )

        info = ""
        for assoc, cnt in loop_closure_counts.items():
            info += f"{assoc}: {cnt} loop closures"

        return info

    @property
    def num_pose_landmark_measurements(self) -> int:
        """Returns the number of pose to landmark measurements.

        Returns:
            int: the number of pose to landmark measurements
        """
        return len(self.pose_landmark_measurements)

    @property
    def num_range_measurements(self) -> int:
        """Returns the number of range measurements.

        Returns:
            int: the number of range measurements
        """
        return len(self.range_measurements)
    
    @property
    def num_bearing_measurements(self) -> int:
        """Returns the number of bearing measurements.

        Returns:
            int: the number of bearing measurements
        """
        return len(self.bearing_measurements)    

    @property
    def num_landmark_priors(self) -> int:
        """Returns the number of landmark priors.

        Returns:
            int: the number of landmark priors
        """
        return len(self.landmark_priors)

    @property
    def num_pose_priors(self) -> int:
        """Returns the number of pose priors.

        Returns:
            int: the number of pose priors
        """
        return len(self.pose_priors)

    @property
    def pose_variables_dict(self) -> Dict[str, POSE_VARIABLE_TYPES]:
        """Returns the pose variables as a dict.

        Returns:
            Dict[str, POSE_VARIABLE_TYPES]: a dict of the pose variables
        """
        pose_var_dict = {}
        for pose_chain in self.pose_variables:
            for pose in pose_chain:
                pose_var_dict[pose.name] = pose
        return pose_var_dict

    @property
    def landmark_variables_dict(self) -> Dict[str, LANDMARK_VARIABLE_TYPES]:
        """Returns the landmark variables as a dict.

        Returns:
            Dict[str, LANDMARK_VARIABLE_TYPES]: a dict of the landmark variables
        """
        landmark_var_dict = {x.name: x for x in self.landmark_variables}
        return landmark_var_dict

    @property
    def variable_true_positions_dict(self) -> Dict[str, Tuple]:
        """Returns the pose and landmark variable true positions as a dict.

        Returns:
            Dict[str, Tuple]: a dict of the pose and landmark variables true positions.
        """
        variable_positions_dict: Dict[str, Tuple] = {}
        pose_variables_dict = self.pose_variables_dict
        for pose_name, pose in pose_variables_dict.items():
            variable_positions_dict[pose_name] = pose.true_position

        landmark_variables_dict = self.landmark_variables_dict
        for var_name, var in landmark_variables_dict.items():
            variable_positions_dict[var_name] = var.true_position

        return variable_positions_dict

    @property
    def all_variable_names(self) -> List[str]:
        """Returns all of the variable names

        Returns:
            List[str]: a list of all the variable names
        """
        var_names = []
        for pose_chain in self.pose_variables:
            for pose in pose_chain:
                var_names.append(pose.name)

        for landmark in self.landmark_variables:
            var_names.append(landmark.name)
        return var_names

    @property
    def unconnected_variable_names(self) -> Set[str]:
        """Returns all of the unconnected variable names

        Returns:
            Set[str]: a set of all the unconnected variable names
        """
        factor_vars: Set[str] = set()
        for odom_chain in self.odom_measurements:
            for odom in odom_chain:
                factor_vars.add(odom.base_pose)
                factor_vars.add(odom.to_pose)

        for pose_landmark_measure in self.pose_landmark_measurements:
            factor_vars.add(pose_landmark_measure.pose_name)
            factor_vars.add(pose_landmark_measure.landmark_name)

        for range_measure in self.range_measurements:
            range_assoc = range_measure.association
            factor_vars.add(range_assoc[0])
            factor_vars.add(range_assoc[1])

        for bearing_measure in self.bearing_measurements:
            bearing_assoc = bearing_measure.association
            factor_vars.add(bearing_assoc[0])
            factor_vars.add(bearing_assoc[1])

        return set(self.all_variable_names) - factor_vars

    @property
    def pose_landmark_measures_association_dict(
        self,
    ) -> Dict[Tuple[str, str], List[POSE_LANDMARK_MEASUREMENT_TYPES]]:
        """Returns a mapping from pose variables to their pose to landmark measurements.

        Returns:
            Dict[Tuple[str, str], List[POSE_LANDMARK_MEASUREMENT_TYPES]]: the mapping from pose variables to their pose to landmark measurements.
        """
        measures_dict: Dict[Tuple[str, str], List[POSE_LANDMARK_MEASUREMENT_TYPES]] = {}
        for measure in self.pose_landmark_measurements:
            association = (measure.pose_name, measure.landmark_name)
            if association not in measures_dict:
                measures_dict[association] = []
            measures_dict[association].append(measure)
        return measures_dict

    # TODO: function is redundant and should be deprecated
    @property
    def pose_to_range_measures_dict(self) -> Dict[str, List[FGRangeMeasurement]]:
        """Returns a mapping from pose variables to their range measurements.

        Returns:
            Dict[str, List[FGRangeMeasurement]]: the mapping from pose variables to their range measurements
        """
        measures_dict: Dict[str, List[FGRangeMeasurement]] = {}
        for measure in self.range_measurements:
            associated_pose = measure.association[0]
            if associated_pose not in measures_dict:
                measures_dict[associated_pose] = []
            measures_dict[associated_pose].append(measure)
        return measures_dict

    # TODO: function is redundant and should be deprecated
    @property
    def both_poses_to_range_measures_dict(self) -> Dict[str, List[FGRangeMeasurement]]:
        """Returns a mapping from both pose variables to their range
        measurements.

        Ex. ("A1", "B3"): [range measurement] -> {"A1": [range measurement], "B3": [range measurement]}

        Returns:
            Dict[str, List[FGRangeMeasurement]]: the mapping from both pose variables to their range measurements
        """
        measures_dict: Dict[str, List[FGRangeMeasurement]] = {}
        for measure in self.range_measurements:
            association_1 = measure.association[0]
            association_2 = measure.association[1]
            if association_1 not in measures_dict:
                measures_dict[association_1] = []
            measures_dict[association_1].append(measure)
            if association_2 not in measures_dict:
                measures_dict[association_2] = []
            measures_dict[association_2].append(measure)
        return measures_dict

    @property
    def range_measures_association_dict(
        self,
    ) -> Dict[Tuple[str, str], List[FGRangeMeasurement]]:
        """Returns a mapping from range measurement associations to their range measurements.

        Returns:
            Dict[Tuple[str, str], List[FGRangeMeasurement]]: the mapping from range measurement associations to their range measurements.
        """
        measures_dict: Dict[Tuple[str, str], List[FGRangeMeasurement]] = {}
        for measure in self.range_measurements:
            association = measure.association
            if association not in measures_dict:
                measures_dict[association] = []
            measures_dict[association].append(measure)
        return measures_dict
    
    # [TODO]: check if pose_to_range_measures_dict for bearing is necessary

    @property
    def bearing_measures_association_dict(
        self,
    ) -> Dict[Tuple[str, str], List[FGBearingMeasurement]]:
        """Returns a mapping from bearing measurement associations to their bearing measurements.

        Returns:
            Dict[Tuple[str, str], List[FGBearingMeasurement]]: the mapping from bearing measurement associations to their bearing measurements.
        """
        measures_dict: Dict[Tuple[str, str], List[FGBearingMeasurement]] = {}
        for measure in self.bearing_measurements:
            association = measure.association
            if association not in measures_dict:
                measures_dict[association] = []
            measures_dict[association].append(measure)
        return measures_dict

    @property
    def true_trajectories(self) -> List[List[np.ndarray]]:
        """Returns the trajectories of ground truth poses.

        Returns:
            List[List[np.ndarray]]: the trajectories of ground truth poses
        """
        true_trajectories = []
        for pose_chain in self.pose_variables:
            true_trajectory = []
            for pose in pose_chain:
                true_trajectory.append(pose.transformation_matrix)
            true_trajectories.append(true_trajectory)
        return true_trajectories

    @property
    def odometry_trajectories(self) -> List[List[np.ndarray]]:
        """Returns the trajectories for each robot obtained from the odometry measurements.

        Returns:
            List[List[np.ndarray]]: the trajectories for each robot obtained from the odometry measurements
        """
        start_poses = [
            pose_chain[0].transformation_matrix for pose_chain in self.pose_variables
        ]
        odom_traj = [[pose] for pose in start_poses]
        for robot_idx, odom_chain in enumerate(self.odom_measurements):
            for odom in odom_chain:
                curr_pose = odom_traj[robot_idx][-1]
                odom_traj[robot_idx].append(curr_pose @ odom.transformation_matrix)
        return odom_traj

    @property
    def odometry_trajectories_dict(self) -> Dict[str, np.ndarray]:
        """Returns the trajectories for each robot obtained from the odometry measurements.

        Returns:
            Dict[str, np.ndarray]: the trajectories for each robot obtained from the odometry measurements
        """
        curr_poses: List[np.ndarray] = [
            np.eye(self.dimension + 1) for _ in range(self.num_robots)
        ]
        odom_traj: Dict[str, np.ndarray] = {}
        for robot_idx, odom_chain in enumerate(self.odom_measurements):
            first_pose_name = odom_chain[0].base_pose
            odom_traj[first_pose_name] = curr_poses[robot_idx]
            for odom in odom_chain:
                curr_poses[robot_idx] = np.dot(
                    curr_poses[robot_idx], odom.transformation_matrix
                )
                pose_name = odom.to_pose
                odom_traj[pose_name] = curr_poses[robot_idx]
        return odom_traj

    @property
    def interrobot_loop_closures(self) -> List[POSE_MEASUREMENT_TYPES]:
        """Returns the loop closure measurements between robots.

        Returns:
            List[POSE_MEASUREMENT_TYPES]: the loop closure measurements between robots
        """
        loop_closures = []
        for measure in self.loop_closure_measurements:
            base_pose_char = measure.base_pose[0]
            to_pose_char = measure.to_pose[0]
            if base_pose_char != to_pose_char:
                loop_closures.append(measure)

        return loop_closures

    @property
    def loop_closure_dict(self) -> Dict[str, List[POSE_MEASUREMENT_TYPES]]:
        """Returns a mapping from pose variables to their loop closure measurements.

        Returns:
            Dict[str, List[POSE_MEASUREMENT_TYPES]]: the mapping from pose variables to their loop closure measurements
        """
        measures_dict: Dict[str, List[POSE_MEASUREMENT_TYPES]] = {}
        for measure in self.loop_closure_measurements:
            associated_pose = measure.base_pose
            if associated_pose not in measures_dict:
                measures_dict[associated_pose] = []
            measures_dict[associated_pose].append(measure)
        return measures_dict

    @property
    def has_priors(self) -> bool:
        """Returns True if the factor graph has priors, and False otherwise.

        Args:
            fg (FactorGraphData): the factor graph data describing the problem

        Returns:
            bool: True if the factor graph has priors, and False otherwise
        """
        has_priors = len(self.pose_priors) > 0 or len(self.landmark_priors) > 0
        return has_priors

    @property
    def all_poses_have_times(self) -> bool:
        """Returns True if all poses have times, and False otherwise.

        Returns:
            bool: True if all poses have times, and False otherwise
        """
        for pose_chain in self.pose_variables:
            for pose in pose_chain:
                if pose.timestamp is None:
                    return False
        return True

    def pose_exists(self, pose_var_name: str) -> bool:
        """Returns whether pose variables exist.

        Args:
            pose_var_name (str): the name of the pose variable

        Returns:
            bool: whether pose variables exist
        """
        return pose_var_name in self.existing_pose_variables

    def landmark_exists(self, landmark_var_name: str) -> bool:
        """Returns whether landmark variables exist.

        Args:
            landmark_var_name (str): the name of the landmark variable

        Returns:
            bool: whether landmark variables exist
        """
        return landmark_var_name in self.existing_landmark_variables

    def is_pose_or_landmark(self, var_name: str) -> bool:
        """Returns whether the variable is a pose or landmark.

        Args:
            var_name (str): the name of the variable

        Returns:
            bool: whether the variable is a pose or landmark
        """
        return self.pose_exists(var_name) or self.landmark_exists(var_name)

    def only_good_measurements(self) -> bool:
        """Checks the measurements for validity.

        Returns:
            bool: whether the measurements are valid
        """
        for odom_chain in self.odom_measurements:
            for odom in odom_chain:
                if odom.translation_precision < 1 or odom.rotation_precision < 1:
                    logger.info(odom)
                    return False

        for pose_landmark_measure in self.pose_landmark_measurements:
            if pose_landmark_measure.translation_precision < 1:
                logger.info(pose_landmark_measure)
                return False

        for range_measure in self.range_measurements:
            if range_measure.weight < 1:
                logger.info(range_measure)
                return False

        for bearing_measure in self.bearing_measurements:
            if bearing_measure.weight < 1:
                logger.info(bearing_measure)
                return False

        return True

    def all_variables_have_factors(self) -> bool:
        """Checks if all variables have factors.

        Returns:
            bool: whether all variables have factors
        """
        return len(self.unconnected_variable_names) == 0

    def get_ranges_by_beacon(self) -> List[List[float]]:
        """Returns the range measurements.

        Returns:
            List[float]: the range measurements
        """
        ranges: List[List[float]] = [[] for _ in range(self.num_landmarks)]
        for range_measure in self.range_measurements:
            if "L" in range_measure.first_key:
                beacon_idx = int(range_measure.first_key[1:])
                ranges[beacon_idx].append(range_measure.dist)

            if "L" in range_measure.second_key:
                beacon_idx = int(range_measure.second_key[1:])
                ranges[beacon_idx].append(range_measure.dist)

        return ranges

    #### Add data

    def _update_max_min_xyz(
        self, var: Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES]
    ) -> None:
        """Helper function for updating the maximum and minimum factor graph positional elements xy (for 2D) and xyz (for 3D)

        Args:
            var (Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES]): variable to update the maximum and minimum factor graph positional elements
        """
        if self.x_min is None or self.x_min > var.true_x:
            self.x_min = var.true_x
        if self.x_max is None or self.x_max < var.true_x:
            self.x_max = var.true_x
        if self.y_min is None or self.y_min > var.true_y:
            self.y_min = var.true_y
        if self.y_max is None or self.y_max < var.true_y:
            self.y_max = var.true_y

        if isinstance(var, PoseVariable3D) or isinstance(var, LandmarkVariable3D):
            if self.z_min is None or self.z_min > var.true_z:
                self.z_min = var.true_z
            if self.z_max is None or self.z_max < var.true_z:
                self.z_max = var.true_z

    def _update_max_min_measurement_weights(
        self, max_weight: float, min_weight: float
    ) -> None:
        """Helper function for updating the maximum and minimum factor graph weights

        Args:
            max_weight (float): the maximum weight
            min_weight (float): the minimum weight
        """
        if self.max_measure_weight is None:
            self.max_measure_weight = max_weight
        elif self.max_measure_weight < max_weight:
            self.max_measure_weight = max_weight

        if self.min_measure_weight is None:
            self.min_measure_weight = min_weight
        elif self.min_measure_weight > min_weight:
            self.min_measure_weight = min_weight

    def add_pose_variable(self, pose_var: POSE_VARIABLE_TYPES) -> None:
        """Adds a pose variable to the list of pose variables.

        Args:
            pose_var (POSE_VARIABLE_TYPES): the pose variable to add

        Raises:
            ValueError: if the pose variable is not added in chronological order
            (time indices must be ordered to ensure that the list is always
            ordered)
        """
        self._check_variable_dimension(pose_var)
        robot_idx = get_robot_idx_from_frame_name(pose_var.name)
        while len(self.pose_variables) <= robot_idx:
            self.pose_variables.append([])

        # enforce that the list is sorted by time index
        new_pose_time_idx = get_time_idx_from_frame_name(pose_var.name)
        if len(self.pose_variables[robot_idx]) > 0:
            last_time_idx = get_time_idx_from_frame_name(
                self.pose_variables[robot_idx][-1].name
            )
            if last_time_idx >= new_pose_time_idx:
                logger.info(self.pose_variables)
                logger.info(pose_var)
                raise ValueError(
                    "Pose variables must be added in order of increasing time_idx"
                )

        self.pose_variables[robot_idx].append(pose_var)
        self.existing_pose_variables.add(pose_var.name)
        self._update_max_min_xyz(pose_var)

    def add_landmark_variable(self, landmark_var: LANDMARK_VARIABLE_TYPES) -> None:
        """Adds a landmark variable to the list of landmark variables.

        Args:
            landmark_var (LANDMARK_VARIABLE_TYPES): the landmark variable to add

        Raises:
            ValueError: if the landmark variable is not added in chronological order
            (time indices must be ordered to ensure that the list is always
            ordered)
        """
        self._check_variable_dimension(landmark_var)
        if len(self.landmark_variables) > 0:
            new_landmark_idx = get_time_idx_from_frame_name(landmark_var.name)
            last_landmark_idx = get_time_idx_from_frame_name(
                self.landmark_variables[-1].name
            )
            if new_landmark_idx <= last_landmark_idx:
                logger.info(self.landmark_variables)
                logger.info(landmark_var)
                raise ValueError(
                    "Landmark variables must be added in order of increasing time_idx"
                )

        self.landmark_variables.append(landmark_var)
        self.existing_landmark_variables.add(landmark_var.name)
        self._update_max_min_xyz(landmark_var)

    def add_odom_measurement(
        self, robot_idx: int, odom_meas: POSE_MEASUREMENT_TYPES
    ) -> None:
        """Adds an odom measurement to the list of odom measurements.

        Args:
            robot_idx (int): the index of the robot that made the measurement
            odom_meas (POSE_MEASUREMENT_TYPES): the odom measurement to add
        """
        self._check_measurement_dimension(odom_meas)
        while len(self.odom_measurements) <= robot_idx:
            self.odom_measurements.append([])

        self.odom_measurements[robot_idx].append(odom_meas)

        # check that we are not adding a measurement between variables that do not exist
        base_pose = odom_meas.base_pose
        assert self.pose_exists(base_pose), f"{base_pose} does not exist"
        to_pose = odom_meas.to_pose
        assert self.pose_exists(to_pose), f"{to_pose} does not exist"

        # update max and min measurement weights
        max_odom_weight = max(
            odom_meas.translation_precision, odom_meas.rotation_precision
        )
        min_odom_weight = min(
            odom_meas.translation_precision, odom_meas.rotation_precision
        )
        self._update_max_min_measurement_weights(max_odom_weight, min_odom_weight)

    def add_loop_closure(self, loop_closure: POSE_MEASUREMENT_TYPES) -> None:
        """Adds a loop closure measurement to the list of loop closure measurements.

        Args:
            loop_closure (POSE_MEASUREMENT_TYPES): the loop closure measurement to add
        """
        self._check_measurement_dimension(loop_closure)
        self.loop_closure_measurements.append(loop_closure)

        # check that we are not adding a measurement between variables that do not exist
        base_pose = loop_closure.base_pose
        assert self.pose_exists(base_pose), f"{base_pose} does not exist"
        to_pose = loop_closure.to_pose
        assert self.pose_exists(to_pose), f"{to_pose} does not exist"

        # update max and min measurement weights
        max_loop_closure_weight = max(
            loop_closure.translation_precision, loop_closure.rotation_precision
        )
        min_loop_closure_weight = min(
            loop_closure.translation_precision, loop_closure.rotation_precision
        )
        self._update_max_min_measurement_weights(
            max_loop_closure_weight, min_loop_closure_weight
        )

    # TODO: Add similar checks to add_odom_measurement and add_loop_closure. Extend function to AmbiguousPoseMeasurement3D and test
    def add_ambiguous_loop_closure(
        self, ambiguous_loop_closure: AmbiguousPoseMeasurement2D
    ) -> None:
        """Adds an ambiguous loop closure measurement to the list of ambiguous loop closure measurements.

        Args:
            measure (AmbiguousPoseMeasurement2D): the ambiguous loop closure measurement to add
        """
        self.ambiguous_loop_closure_measurements.append(ambiguous_loop_closure)

    def add_pose_landmark_measurement(
        self, pose_landmark_meas: POSE_LANDMARK_MEASUREMENT_TYPES
    ) -> None:
        """Adds a pose to landmark measurement to the list of pose to landmark measurements.

        Args:
            pose_landmark_meas (POSE_MEASUREMENT_TYPES): the pose to landmark measurement to add
        """
        self._check_measurement_dimension(pose_landmark_meas)
        self.pose_landmark_measurements.append(pose_landmark_meas)

        # check that we are not adding a measurement between variables that do not exist
        pose_name = pose_landmark_meas.pose_name
        assert self.pose_exists(pose_name), f"{pose_name} does not exist"
        landmark_name = pose_landmark_meas.landmark_name
        assert self.landmark_exists(landmark_name), f"{landmark_name} does not exist"

        # update max and min measurement weights
        pose_landmark_weight = pose_landmark_meas.translation_precision
        self._update_max_min_measurement_weights(
            pose_landmark_weight, pose_landmark_weight
        )

    def add_range_measurement(self, range_meas: FGRangeMeasurement) -> None:
        """Adds a range measurement to the list of range measurements.

        Args:
            range_meas (FGRangeMeasurement): the range measurement to add
        """

        # check that we are not adding a measurement between variables that exist
        var1, var2 = range_meas.association
        assert self.is_pose_or_landmark(var1)
        assert self.is_pose_or_landmark(var2)
        self.range_measurements.append(range_meas)

        # update max and min measurement weights
        if (
            self.max_measure_weight is None
            or self.max_measure_weight < range_meas.weight
        ):
            self.max_measure_weight = range_meas.weight

        if (
            self.min_measure_weight is None
            or self.min_measure_weight > range_meas.weight
        ):
            self.min_measure_weight = range_meas.weight

    # TODO: implement similar checks to add_range_measurement
    def add_ambiguous_range_measurement(
        self, measure: AmbiguousFGRangeMeasurement
    ) -> None:
        """Adds an ambiguous range measurement to the list of ambiguous range measurements.

        Args:
            measure (AmbiguousFGRangeMeasurement): the ambiguous range measurement to add
        """
        self.ambiguous_range_measurements.append(measure)

    def add_bearing_measurement(self, bearing_meas: FGBearingMeasurement) -> None:
        """Adds a bearing measurement to the list of bearing measurements.

        Args:
            bearing_meas (FGBearingMeasurement): the bearing measurement to add
        """

        # check that we are not adding a measurement between variables that exist
        var1, var2 = bearing_meas.association
        assert self.is_pose_or_landmark(var1)
        assert self.is_pose_or_landmark(var2)
        self.bearing_measurements.append(bearing_meas)

        # update max and min measurement weights
        if (
            self.max_measure_weight is None
            or self.max_measure_weight < bearing_meas.weight
        ):
            self.max_measure_weight = bearing_meas.weight

        if (
            self.min_measure_weight is None
            or self.min_measure_weight > bearing_meas.weight
        ):
            self.min_measure_weight = bearing_meas.weight

    def add_pose_prior(self, pose_prior: POSE_PRIOR_TYPES) -> None:
        """Adds a pose prior to the list of pose priors.

        Args:
            pose_prior (POSE_PRIOR_TYPES): the pose prior to add
        """
        self._check_prior_dimension(pose_prior)
        self.pose_priors.append(pose_prior)

    def add_landmark_prior(self, landmark_prior: LANDMARK_PRIOR_TYPES) -> None:
        """Adds a landmark prior to the list of landmark priors.

        Args:
            landmark_prior (LANDMARK_PRIOR_TYPES): the landmark prior to add
        """
        self._check_prior_dimension(landmark_prior)
        self.landmark_priors.append(landmark_prior)

    #### Get pose chain variable names

    def get_pose_chain_names(self) -> List[List[str]]:
        """Returns the pose chain variable names.

        Returns:
            List[str]: the pose chain variable names
        """
        pose_chain_names = []
        for pose_chain in self.pose_variables:
            pose_chain_names.append([pose.name for pose in pose_chain])
        return pose_chain_names

    #### saving functionalities

    # TODO: deprecate for pyfg_text.py
    def save_to_file(self, filepath: str):
        """
        Save the factor graph to a file. The format is determined by the file
        extension. We currently support ["fg", "pickle"].

        Args:
            filepath (str): the path of the file to write to
        """
        file_dir = os.path.dirname(filepath)
        assert os.path.isdir(file_dir), f"{file_dir} is not a directory"

        # check is valid file type
        file_extension = pathlib.Path(filepath).suffix.strip(".")
        format_options = ["fg", "pickle", "plaza", "pkl"]
        assert (
            file_extension in format_options
        ), f"File extension: {file_extension} not available, must be one of {format_options}"

        if file_extension == "fg":
            self._save_to_efg_format(filepath)
        elif file_extension in ["pickle", "pkl"]:
            self._save_to_pickle_format(filepath)
        elif file_extension == "plaza":
            self._save_to_plaza_format(file_dir)
        else:
            raise ValueError(f"Unknown format: {file_extension}")

        logger.info(f"Saved data to {filepath}")

    # TODO: deprecate for pyfg_text.py
    def _save_to_efg_format(
        self,
        data_file: str,
    ) -> None:
        """
        Save the given data to the extended factor graph format.

        Args:
            data_file (str): the path of the file to write to
        """

        assert self.dimension == 2, "Only 2D factor graphs are supported for EFG format"

        def get_normal_pose_measurement_string(pose_measure: PoseMeasurement2D) -> str:
            """This is a utility function to get a formatted string to write to EFG
            formats for measurements which can be represented by poses (i.e.
            odometry and loop closures.

            Args:
                pose (PoseMeasurement2D): the measurement

            Returns:
                str: the formatted string representation of the pose measurement
            """
            line = "Factor SE2RelativeGaussianLikelihoodFactor "

            base_key = pose_measure.base_pose
            to_key = pose_measure.to_pose
            line += f"{base_key} {to_key} "

            # add in odometry info
            del_x = pose_measure.x
            del_y = pose_measure.y
            del_theta = pose_measure.theta
            line += f"{del_x:.15f} {del_y:.15f} {del_theta:.15f} "

            # add in covariance info
            line += "covariance "
            covar_info = pose_measure.covariance.flatten()
            for val in covar_info:
                line += f"{val:.15f} "

            line += "\n"
            # return the formatted string
            return line

        def get_ambiguous_pose_measurement_string(
            pose_measure: AmbiguousPoseMeasurement2D,
        ) -> str:
            """This is a utility function to get a formatted string to write to EFG
            formats for measurements which can be represented by poses (i.e.
            odometry and loop closures.

            Args:
                pose (SE2Pose): the measurement

            Returns:
                str: the formatted string representation of the pose measurement
            """
            line = "Factor AmbiguousDataAssociationFactor "

            cur_id = pose_measure.base_pose
            line += f"Observer {cur_id} "

            true_measure_id = pose_measure.true_to_pose
            measure_id = pose_measure.measured_to_pose
            line += f"Observed {true_measure_id} {measure_id} "
            line += "Weights 0.5 0.5 Binary SE2RelativeGaussianLikelihoodFactor Observation "

            # add in odometry info
            del_x = pose_measure.x
            del_y = pose_measure.y
            del_theta = pose_measure.theta
            line += f"{del_x:.15f} {del_y:.15f} {del_theta:.15f} "

            # add in covariance info (Sigma == covariance)
            line += "Sigma "
            covar_info = pose_measure.covariance.flatten()
            for val in covar_info:
                line += f"{val:.15f} "

            line += "\n"
            # return the formatted string
            return line

        def get_pose_var_string(pose: PoseVariable2D) -> str:
            """
            Takes a pose and returns a string in the desired format
            """
            line = "Variable Pose SE2 "

            # get local frame for pose
            pose_key = pose.name

            # add in pose information
            line += f"{pose_key} {pose.true_x:.15f} {pose.true_y:.15f} {pose.true_theta:.15f}\n"

            return line

        def get_beacon_var_string(beacon: LandmarkVariable2D) -> str:
            """Takes in a beacon and returns a string formatted as desired

            Args:
                beacon (Beacon): the beacon

            Returns:
                str: the formatted string
            """
            line = "Variable Landmark R2 "

            frame = beacon.name
            line += f"{frame} {beacon.true_x:.15f} {beacon.true_y:.15f}\n"
            return line

        def get_prior_to_pin_string(prior: PosePrior2D) -> str:
            """this is the prior on the first pose to 'pin' the factor graph.

            Returns:
                str: the line representing the prior
            """
            prior_key = prior.name
            line = f"Factor UnarySE2ApproximateGaussianPriorFactor {prior_key} "
            line += f"{prior.x:.15f} {prior.y:.15f} {prior.theta:.15f} "

            line += "covariance "
            cov = prior.covariance.flatten()
            for val in cov:
                line += f"{val:.15f} "
            line += "\n"

            return line

        def get_range_measurement_string(
            range_measure: FGRangeMeasurement,
        ) -> str:
            """Returns the string representing a range factor based on the provided
            range measurement and the association information.

            Args:
                range_measure (FGRangeMeasurement): the measurement info (value and
                    stddev)

            Returns:
                str: the line representing the factor
            """

            robot_id, measure_id = range_measure.association

            # Factor SE2R2RangeGaussianLikelihoodFactor X0 L1 14.14214292904807 0.5
            if "L" in measure_id:
                # L is reserved for landmark names
                range_factor_type = "SE2R2RangeGaussianLikelihoodFactor"
            else:
                # ID starts with other letters are robots
                range_factor_type = "SE2SE2RangeGaussianLikelihoodFactor"
            line = f"Factor {range_factor_type} "
            line += f"{robot_id} {measure_id} "
            line += f"{range_measure.dist:.15f} {range_measure.stddev:.15f}\n"

            return line

        def get_ambiguous_range_measurement_string(
            range_measure: AmbiguousFGRangeMeasurement,
        ) -> str:
            """Returns the string representing an ambiguous range factor based on
            the provided range measurement and the association information.

            Args:
                range_measure (AmbiguousFGRangeMeasurement): the measurement info
                    (value and stddev)

            Returns:
                str: the line representing the factor
            """

            true_robot_id, true_beacon_id = range_measure.true_association
            measured_robot_id, measured_beacon_id = range_measure.measured_association

            assert (
                true_robot_id == measured_robot_id
            ), "the robot id must always be correct"
            assert (
                "L" in true_beacon_id and "L" in measured_beacon_id
            ), "right now only considering ambiguous measurements to landmarks"

            # Factor AmbiguousDataAssociationFactor Observer X1 Observed L1 L2
            # Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor
            # Observation 24.494897460297107 Sigma 0.5
            line = "Factor AmbiguousDataAssociationFactor "
            line += f"Observer {true_robot_id} "
            line += f"Observed {true_beacon_id} {measured_beacon_id} "
            line += "Weights 0.5 0.5 Binary SE2R2RangeGaussianLikelihoodFactor "
            line += f"Observation {range_measure.dist:.15f} Sigma {range_measure.stddev:.15f}\n"

            return line
        
        def get_bearing_measurement_string(
            bearing_measure: FGBearingMeasurement,
        ) -> str:
            """Returns the string representing a bearing factor based on the provided
            bearing measurement and the association information.

            Args:
                bearing_measure (FGBearingMeasurement): the measurement info (value and
                    stddev)

            Returns:
                str: the line representing the factor
            """

            robot_id, measure_id = bearing_measure.association

            # [TODO]: Double check this
            bearing_factor_type = "SE2BearingLikelihoodFactor"
            line = f"Factor {bearing_factor_type} "
            line += f"{robot_id} {measure_id} "
            line += f"{bearing_measure.bearing_azimuth:.15f} {bearing_measure.bearing_elevation:.15f} {bearing_measure.azimuth_stddev:.15f} {bearing_measure.elevation_stddev:.15f}\n"

            return line

        file_writer = open(data_file, "w")

        for pose_chain in self.pose_variables:
            for pose in pose_chain:
                assert isinstance(pose, PoseVariable2D)
                line = get_pose_var_string(pose)
                file_writer.write(line)

        for beacon in self.landmark_variables:
            assert isinstance(beacon, LandmarkVariable2D)
            line = get_beacon_var_string(beacon)
            file_writer.write(line)

        for prior in self.pose_priors:
            assert isinstance(
                prior, PosePrior2D
            ), "3D priors not supported yet for efg format"
            line = get_prior_to_pin_string(prior)
            file_writer.write(line)

        for odom_chain in self.odom_measurements:
            for odom_measure in odom_chain:
                assert isinstance(odom_measure, PoseMeasurement2D)
                line = get_normal_pose_measurement_string(odom_measure)
                file_writer.write(line)

        for loop_closure in self.loop_closure_measurements:
            assert isinstance(loop_closure, PoseMeasurement2D)
            line = get_normal_pose_measurement_string(loop_closure)
            file_writer.write(line)

        for amb_odom_measure in self.ambiguous_loop_closure_measurements:
            assert isinstance(amb_odom_measure, AmbiguousPoseMeasurement2D)
            line = get_ambiguous_pose_measurement_string(amb_odom_measure)
            file_writer.write(line)

        for range_measure in self.range_measurements:
            assert isinstance(range_measure, FGRangeMeasurement)
            line = get_range_measurement_string(range_measure)
            file_writer.write(line)

        for amb_range_measure in self.ambiguous_range_measurements:
            assert isinstance(amb_range_measure, AmbiguousFGRangeMeasurement)
            line = get_ambiguous_range_measurement_string(amb_range_measure)
            file_writer.write(line)

        for bearing_measure in self.bearing_measurements:
            assert isinstance(bearing_measure, FGBearingMeasurement)
            line = get_bearing_measurement_string(bearing_measure)
            file_writer.write(line)

        file_writer.close()

    # TODO: deprecate for pyfg_text.py
    def _save_to_pickle_format(self, data_file: str) -> None:
        """
        Save to pickle format.
        """
        file_dir = os.path.dirname(data_file)
        if not os.path.exists(file_dir):
            logger.info(f"Creating directory {file_dir}")
            os.makedirs(file_dir)

        pickle_file = open(data_file, "wb")
        pickle.dump(self, pickle_file)
        pickle_file.close()

    # TODO: deprecate for pyfg_text.py. We should instead have an io parser that converts between pyfg_text and plaza_format
    def _save_to_plaza_format(self, data_folder: str) -> None:
        """
        Save to five plaza file formats.

        Args:
            data_folder (str): the base folder to write the files to
        """
        assert (
            len(self.pose_variables) == 1
        ), ".plaza file format only supports one robot"
        assert self.dimension == 2, ".plaza file format only supports 2D"

        def save_GT_plaza() -> None:
            """
            Save Ground Truth plaza file
            """
            filename = data_folder + "/GT.plaza"
            filewriter = open(filename, "w")

            for pose in self.pose_variables[0]:
                assert isinstance(pose, PoseVariable2D)
                line = f"{pose.timestamp} {pose.true_position[0]} {pose.true_position[1]} {pose.true_theta}"
                filewriter.write(line)

            filewriter.close()

        def save_DR_plaza() -> None:
            """
            Save Odometry Input plaza file
            """
            filename = data_folder + "/DR.plaza"
            filewriter = open(filename, "w")

            for odom in self.odom_measurements[0]:
                # We only take odom.x because plaza assumes we are moving in the
                # direction of the robot's heading
                assert isinstance(odom, PoseMeasurement2D)
                line = f"{odom.timestamp} {odom.x} {odom.theta}"
                filewriter.write(line)

            filewriter.close()

        def save_DRp_plaza() -> None:
            """
            Save Dead Reckoning Path from Odometry plaza file
            """
            filename = data_folder + "/DRp.plaza"
            filewriter = open(filename, "w")

            init_pose = self.pose_variables[0][0]
            dr_pose = init_pose.transformation_matrix

            for odom in self.odom_measurements[0]:
                assert isinstance(odom, PoseMeasurement2D)
                dr_pose = dr_pose @ odom.transformation_matrix
                dr_theta = get_theta_from_transformation_matrix(dr_pose)
                line = f"{odom.timestamp} {dr_pose[0,2]} {dr_pose[1,2]} {dr_theta}"
                filewriter.write(line)

            filewriter.close()

        def save_TL_plaza() -> None:
            """
            Save Surveyed Node Locations plaza file
            """
            filename = data_folder + "/TL.plaza"
            filewriter = open(filename, "w")

            for landmark in self.landmark_variables:
                line = f"{0} {landmark.true_x} {landmark.true_y}"
                filewriter.write(line)

            filewriter.close()

        def save_TD_plaza() -> None:
            """
            Save Range Measurements plaza file
            """
            filename = data_folder + "/TD.plaza"
            filewriter = open(filename, "w")

            for range_measurement in self.range_measurements:
                line = f"{range_measurement.timestamp} {range_measurement.first_key} {range_measurement.second_key} {range_measurement.dist}"
                filewriter.write(line)

            filewriter.close()

        def save_TB_plaza() -> None:
            """
            Save Bearing Measurements plaza file
            """
            filename = data_folder + "/TD.plaza"
            filewriter = open(filename, "w")

            for bearing_measurement in self.bearing_measurements:
                line = f"{bearing_measurement.timestamp} {bearing_measurement.first_key} {bearing_measurement.second_key} {bearing_measurement.bearing_azimuth} {bearing_measurement.bearing_elevation}"
                filewriter.write(line)

            filewriter.close()

        save_GT_plaza()
        save_DR_plaza()
        save_DRp_plaza()
        save_TL_plaza()
        save_TD_plaza()
        save_TB_plaza()
        return

    # TODO: deprecate for pyfg_text.py. We should instead have an io parser that converts between pyfg_text and tum
    def write_pose_gt_to_tum(self, data_dir: str) -> List[str]:
        """
        Write ground truth to TUM format.

        Args:
            data_dir (str): the base folder to write the files to

        Returns:
            List[str]: the list of file paths written
        """
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        logger.debug(f"Writing ground truth to TUM format in {data_dir}")
        gt_files = []
        possible_end_chars = [chr(ord("A") + i) for i in range(26)]
        # remove 'L' from possible end chars because it is reserved for landmarks
        possible_end_chars.remove("L")
        for i, pose_chain in enumerate(self.pose_variables):
            filename = f"gt_traj_{possible_end_chars[i]}.tum"
            filepath = os.path.join(data_dir, filename)
            fw = open(filepath, "w")

            for pose_idx, pose in enumerate(pose_chain):
                timestamp = pose.timestamp if pose.timestamp is not None else pose_idx
                qx, qy, qz, qw = pose.true_quat
                fw.write(
                    f"{timestamp} "
                    f"{pose.true_x} {pose.true_y} {pose.true_z} "
                    f"{qx} {qy} {qz} {qw}\n"
                )

            fw.close()
            logger.debug(f"Saved to {filepath}")
            gt_files.append(filepath)

        return gt_files

    # TODO: deprecate for pyfg_text.py. We should instead have an io parser that converts between pyfg_text and tum
    def write_pose_odom_to_tum(self, data_dir: str) -> List[str]:
        """Write odometry to TUM format.

        Args:
            data_dir (str): the base folder to write the files to

        Returns:
            List[str]: the list of file paths written
        """
        logger.debug(f"Writing odometry to TUM format in {data_dir}")
        odom_files = []
        for i, odom_chain in enumerate(self.odom_measurements):
            filename = "odom_traj_" + chr(ord("A") + i) + ".tum"
            filepath = os.path.join(data_dir, filename)
            fw = open(filepath, "w")

            start_pose = self.pose_variables[i][0].transformation_matrix
            start_timestamp = (
                self.pose_variables[i][0].timestamp
                if self.pose_variables[i][0].timestamp is not None
                else 0
            )
            x, y, z = get_translation_from_transformation_matrix(start_pose)
            rot = get_rotation_matrix_from_transformation_matrix(start_pose)
            qx, qy, qz, qw = get_quat_from_rotation_matrix(rot)
            fw.write(f"{start_timestamp} " f"{x} {y} {z} " f"{qx} {qy} {qz} {qw}\n")

            cur_pose = start_pose
            for odom_idx, odom in enumerate(odom_chain):
                odom_mat = odom.transformation_matrix
                cur_pose = cur_pose @ odom_mat

                cur_x, cur_y, cur_z = get_translation_from_transformation_matrix(
                    cur_pose
                )
                cur_rot = get_rotation_matrix_from_transformation_matrix(cur_pose)
                cur_qx, cur_qy, cur_qz, cur_qw = get_quat_from_rotation_matrix(cur_rot)

                timestamp = (
                    odom.timestamp if odom.timestamp is not None else (odom_idx + 1)
                )
                fw.write(
                    f"{timestamp} "
                    f"{cur_x} {cur_y} {cur_z} "
                    f"{cur_qx} {cur_qy} {cur_qz} {cur_qw}\n"
                )

            fw.close()
            logger.info(f"Saved to {filepath}")
            odom_files.append(filepath)

        return odom_files

    #### plotting functions ####

    # TODO: move to io
    def plot_odom_precisions(self) -> None:
        """
        Plot the translation and rotation precisions on two separate
        subplots.
        """
        trans_precisions = []
        rot_precisions = []
        for odom_chain in self.odom_measurements:
            for odom_measure in odom_chain:
                trans_precisions.append(odom_measure.translation_precision)
                rot_precisions.append(odom_measure.rotation_precision)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # plot translation precisions on top subplot
        axs[0].plot(trans_precisions)

        # plot rotation precisions on bottom
        axs[1].plot(rot_precisions)

        plt.show(block=True)

    # TODO: move to io
    def plot_ranges(self) -> None:
        num_beacons = self.num_landmarks
        ranges = self.get_ranges_by_beacon()
        fig, axs = plt.subplots(num_beacons, 1, figsize=(10, 10))

        max_range = 0.0
        for idx in range(num_beacons):
            if len(ranges[idx]) > 0:
                max_range = max(max(ranges[idx]), max_range)
            axs[idx].plot(ranges[idx])

        for idx in range(num_beacons):
            axs[idx].set_ylim([0, max_range])

        plt.show(block=True)

    # TODO: move to io
    def animate_odometry(
        self,
        show_gt: bool = False,
        pause_interval: float = 0.01,
        draw_range_lines: bool = False,
        draw_range_circles: bool = False,
        num_timesteps_keep_ranges: int = 1,
    ) -> None:
        """Makes an animation of the odometric chain for every robot

        Args:
            show_gt (bool, optional): whether to show the ground truth as well. Defaults to False.
            pause (float, optional): How long to pause between frames. Defaults to 0.01.
        """
        assert self.dimension == 2, "Only 2D data can be animated"
        assert (
            self.x_min is not None and self.x_max is not None
        ), "x_min and x_max must be set"
        assert (
            self.y_min is not None and self.y_max is not None
        ), "y_min and y_max must be set"

        # set up plot
        fig, ax = plt.subplots()

        # scale is the order of magnitude of the largest range
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        scale = max(x_range, y_range) / 100.0

        def _format_viewport():
            x_min = self.x_min - 0.1 * abs(self.x_min)
            x_max = self.x_max + 0.1 * abs(self.x_max)
            y_min = self.y_min - 0.1 * abs(self.y_min)
            y_max = self.y_max + 0.1 * abs(self.y_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # set axes to be equal
            ax.set_aspect("equal")

        _format_viewport()

        # go ahead and draw the landmarks
        for landmark in self.landmark_variables:
            assert isinstance(landmark, LandmarkVariable2D)
            draw_landmark_variable(ax, landmark)

        # all of the objects to track for plotting the poses as arrows
        odom_color = "blue"
        gt_color = "red"
        odom_pose_var_arrows: List[mpatches.FancyArrow] = [
            draw_pose(ax, pose_chain[0], odom_color, scale=scale) for pose_chain in self.pose_variables  # type: ignore
        ]
        gt_pose_var_arrows: List[mpatches.FancyArrow] = [
            draw_pose(ax, pose_chain[0], gt_color, scale=scale) for pose_chain in self.pose_variables  # type: ignore
        ]

        # all of the objects to track for plotting the trajectories as lines
        odom_pose_traj_lines: List[mlines.Line2D] = [
            draw_traj(
                ax,
                [],
                [],
                odom_color,
            )
            for robot_idx in range(self.num_robots)
        ]
        gt_pose_traj_lines: List[mlines.Line2D] = [
            draw_traj(
                ax,
                [],
                [],
                gt_color,
            )
            for _ in range(self.num_robots)
        ]

        # get the trajectories as poses
        odom_pose_trajs = self.odometry_trajectories
        gt_pose_trajs = self.true_trajectories
        assert len(odom_pose_trajs) == len(
            gt_pose_trajs
        ), "must be same number of odometry and ground truth pose chains"

        def _get_xy_traj_from_pose_traj(
            pose_traj: List[np.ndarray],
        ) -> Tuple[List[float], List[float]]:
            x_traj = [pose[0, 2] for pose in pose_traj]
            y_traj = [pose[1, 2] for pose in pose_traj]
            return x_traj, y_traj

        odom_xy_history = [
            _get_xy_traj_from_pose_traj(pose_traj) for pose_traj in odom_pose_trajs
        ]
        gt_pose_xy_history = [
            _get_xy_traj_from_pose_traj(pose_traj) for pose_traj in gt_pose_trajs
        ]

        num_timesteps = len(odom_pose_trajs[0])

        def _update_traj_lines(timestep: int) -> None:
            idx = min(timestep + 1, num_timesteps)
            for robot_idx in range(self.num_robots):
                odom_pose_traj_lines[robot_idx].set_data(
                    odom_xy_history[robot_idx][0][:idx],
                    odom_xy_history[robot_idx][1][:idx],
                )
                if show_gt:
                    gt_pose_traj_lines[robot_idx].set_data(
                        gt_pose_xy_history[robot_idx][0][:idx],
                        gt_pose_xy_history[robot_idx][1][:idx],
                    )

        def _update_pose_arrows(timestep: int) -> None:
            for robot_idx in range(self.num_robots):
                update_pose_arrow(
                    odom_pose_var_arrows[robot_idx],
                    odom_pose_trajs[robot_idx][timestep],
                    scale=scale,
                )
                if show_gt:
                    update_pose_arrow(
                        gt_pose_var_arrows[robot_idx],
                        gt_pose_trajs[robot_idx][timestep],
                        scale=scale,
                    )

        pose_range_measures = self.pose_to_range_measures_dict
        pose_dict = self.pose_variables_dict
        landmark_dict = self.landmark_variables_dict
        range_line_drawings: List[mlines.Line2D] = []
        range_circle_drawings: List[mpatches.Circle] = []
        range_timesteps_added: List[int] = []  # keep track of when we added the range

        def _update_range_lines(timestep: int) -> None:
            def _has_range_measures_to_remove():
                return (
                    len(range_timesteps_added) > 0
                    and timestep - range_timesteps_added[0] > num_timesteps_keep_ranges
                )

            while _has_range_measures_to_remove():
                drawn_line = range_line_drawings.pop(0)
                drawn_circle = range_circle_drawings.pop(0)
                range_timesteps_added.pop(0)

                if drawn_line is not None:
                    drawn_line.remove()
                if drawn_circle is not None:
                    drawn_circle.remove()

            for robot_idx in range(self.num_robots):
                pose = self.pose_variables[robot_idx][timestep]
                assert isinstance(pose, PoseVariable2D)
                if pose.name not in pose_range_measures:
                    continue

                associated_range_measures = pose_range_measures[pose.name]
                for range_measure in associated_range_measures:
                    other_var_name = range_measure.association[1]
                    if other_var_name in landmark_dict:
                        other_var = landmark_dict[other_var_name]  # type: ignore
                    elif other_var_name in pose_dict:
                        other_var = pose_dict[other_var_name]  # type: ignore

                    assert isinstance(other_var, (PoseVariable2D, LandmarkVariable2D))
                    drawn_line, drawn_circle = draw_range_measurement(
                        ax,
                        range_measure,
                        pose,
                        other_var,
                        add_line=draw_range_lines,
                        add_circle=draw_range_circles,
                    )
                    range_line_drawings.append(drawn_line)
                    range_circle_drawings.append(drawn_circle)
                    range_timesteps_added.append(timestep)

        def _update_animation(timestep: int) -> None:
            _update_traj_lines(timestep)
            _update_pose_arrows(timestep)
            if draw_range_circles or draw_range_lines:
                _update_range_lines(timestep)

        # add a legend for blue = odometry, red = ground truth
        legend_elements = [
            mlines.Line2D([0], [0], color=odom_color, label="odometry"),
            mlines.Line2D([0], [0], color=gt_color, label="ground truth"),
        ]
        ax.legend(handles=legend_elements)

        ani = animation.FuncAnimation(
            fig,
            _update_animation,
            frames=num_timesteps,
            interval=pause_interval,
            repeat=False,
        )

        plt.show(block=True)

    # TODO: move to io
    def animate_odometry_3d(
        self,
        show_gt: bool = False,
        pause_interval: float = 0.01,
        draw_range_lines: bool = False,
        num_timesteps_keep_ranges: int = 1,
    ) -> None:
        """Makes an animation of the odometric chain for every robot

        Args:
            show_gt (bool, optional): whether to show the ground truth as well. Defaults to False.
            pause (float, optional): How long to pause between frames. Defaults to 0.01.
        """
        assert self.dimension == 3, "Only 3D data can be animated"
        assert (
            self.x_min is not None and self.x_max is not None
        ), "x_min and x_max must be set"
        assert (
            self.y_min is not None and self.y_max is not None
        ), "y_min and y_max must be set"
        assert (
            self.z_min is not None and self.z_max is not None
        ), "z_min and z_max must be set"

        # set up plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # scale is the order of magnitude of the largest range
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        z_range = self.z_max - self.z_min
        scale = max(x_range, y_range, z_range) / 100.0

        def _format_viewport():
            x_min = self.x_min - 0.1 * abs(self.x_min)
            x_max = self.x_max + 0.1 * abs(self.x_max)
            y_min = self.y_min - 0.1 * abs(self.y_min)
            y_max = self.y_max + 0.1 * abs(self.y_max)
            z_min = self.z_min - 0.1 * abs(self.z_min)
            z_max = self.z_max + 0.1 * abs(self.z_max)
            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)

            # set axes to be equal
            ax.set_aspect("auto")

        _format_viewport()

        # go ahead and draw the landmarks
        for landmark in self.landmark_variables:
            assert isinstance(landmark, LandmarkVariable3D)
            draw_landmark_variable_3d(ax, landmark)

        # all of the objects to track for plotting the poses as arrows
        odom_color = "blue"
        gt_color = "red"

        # all of the objects to track for plotting the trajectories as lines
        odom_pose_traj_lines: List[art3d.Line3D] = [
            draw_traj_3d(
                ax,
                [],
                [],
                [],
                odom_color,
            )
            for robot_idx in range(self.num_robots)
        ]
        gt_pose_traj_lines: List[art3d.Line3D] = [
            draw_traj_3d(
                ax,
                [],
                [],
                [],
                gt_color,
            )
            for _ in range(self.num_robots)
        ]

        # get the trajectories as poses
        odom_pose_trajs = self.odometry_trajectories
        gt_pose_trajs = self.true_trajectories
        assert len(odom_pose_trajs) == len(
            gt_pose_trajs
        ), "must be same number of odometry and ground truth pose chains"

        def _get_xyz_traj_from_pose_traj(
            pose_traj: List[np.ndarray],
        ) -> Tuple[List[float], List[float], List[float]]:
            x_traj = [pose[0, 3] for pose in pose_traj]
            y_traj = [pose[1, 3] for pose in pose_traj]
            z_traj = [pose[2, 3] for pose in pose_traj]
            return x_traj, y_traj, z_traj

        odom_xyz_history = [
            _get_xyz_traj_from_pose_traj(pose_traj) for pose_traj in odom_pose_trajs
        ]
        gt_pose_xyz_history = [
            _get_xyz_traj_from_pose_traj(pose_traj) for pose_traj in gt_pose_trajs
        ]

        num_timesteps = len(odom_pose_trajs[0])

        def _update_traj_lines(timestep: int) -> None:
            idx = min(timestep + 1, num_timesteps)
            for robot_idx in range(self.num_robots):
                odom_pose_traj_lines[robot_idx].set_data_3d(
                    odom_xyz_history[robot_idx][0][:idx],
                    odom_xyz_history[robot_idx][1][:idx],
                    odom_xyz_history[robot_idx][2][:idx],
                )
                if show_gt:
                    gt_pose_traj_lines[robot_idx].set_data_3d(
                        gt_pose_xyz_history[robot_idx][0][:idx],
                        gt_pose_xyz_history[robot_idx][1][:idx],
                        gt_pose_xyz_history[robot_idx][2][:idx],
                    )

        prev_arrows: List[art3d.Line3DCollection] = []

        def _update_pose_arrows(timestep: int) -> None:
            for arrow in prev_arrows:
                arrow.remove()
            prev_arrows.clear()

            for robot_idx in range(self.num_robots):
                arrow = draw_pose_3d(
                    ax,
                    odom_pose_trajs[robot_idx][timestep],
                    color=odom_color,
                    scale=scale,
                )
                prev_arrows.append(arrow)

                if show_gt:
                    arrow = draw_pose_3d(
                        ax,
                        gt_pose_trajs[robot_idx][timestep],
                        color=gt_color,
                        scale=scale,
                    )
                    prev_arrows.append(arrow)

        pose_range_measures = self.pose_to_range_measures_dict
        pose_dict = self.pose_variables_dict
        landmark_dict = self.landmark_variables_dict
        range_line_drawings: List[art3d.Line3D] = []
        range_timesteps_added: List[int] = []  # keep track of when we added the range

        def _update_range_lines(timestep: int) -> None:
            def _has_range_measures_to_remove():
                return (
                    len(range_timesteps_added) > 0
                    and timestep - range_timesteps_added[0] > num_timesteps_keep_ranges
                )

            while _has_range_measures_to_remove():
                drawn_line = range_line_drawings.pop(0)
                range_timesteps_added.pop(0)

                if drawn_line is not None:
                    drawn_line.remove()

            for robot_idx in range(self.num_robots):
                pose = self.pose_variables[robot_idx][timestep]
                assert isinstance(pose, PoseVariable3D)
                if pose.name not in pose_range_measures:
                    continue

                associated_range_measures = pose_range_measures[pose.name]
                for range_measure in associated_range_measures:
                    other_var_name = range_measure.association[1]
                    if other_var_name in landmark_dict:
                        other_var = landmark_dict[other_var_name]  # type: ignore
                    elif other_var_name in pose_dict:
                        other_var = pose_dict[other_var_name]  # type: ignore

                    assert isinstance(other_var, (PoseVariable3D, LandmarkVariable3D))
                    drawn_line = draw_range_measurement_3d(
                        ax,
                        range_measure,
                        pose,
                        other_var,
                        add_line=draw_range_lines,
                    )
                    range_line_drawings.append(drawn_line)
                    range_timesteps_added.append(timestep)

        def _update_animation(timestep: int) -> None:
            _update_traj_lines(timestep)
            _update_pose_arrows(timestep)
            if draw_range_lines:
                _update_range_lines(timestep)

        # add a legend for blue = odometry, red = ground truth
        legend_elements = [
            art3d.Line3D([0], [0], [0], color=odom_color, label="odometry"),
            art3d.Line3D([0], [0], [0], color=gt_color, label="ground truth"),
        ]
        ax.legend(handles=legend_elements)

        ani = animation.FuncAnimation(
            fig,
            _update_animation,
            frames=num_timesteps,
            interval=pause_interval,
            repeat=False,
        )

        plt.show(block=True)

    #### checks on inputs ####

    def _dimension_logger(self, is_2d: bool, is_3d: bool, pyfg_type: type) -> None:
        """Checks the correct dimension against is_2d and is_3d corresponding to pyfg_type

        Args:
            is_2d (bool): True if 2D, and False otherwise
            is_3d (bool): True if 3D, and False otherwise
            pyfg_type: the variable or measurement type corresponding to is_2d and is_3d

        Raises:
            ValueError: if the pyfg_type is not the correct dimension
        """
        if not (is_2d or is_3d):
            raise ValueError(f"Variable must be either 2D or 3D, but got {pyfg_type}")

        if is_2d and self.dimension != 2:
            raise ValueError(
                f"Variable is 2D but the dimension of the graph is {self.dimension}"
            )

        if is_3d and self.dimension != 3:
            raise ValueError(
                f"Variable is 3D but the dimension of the graph is {self.dimension}"
            )

    # TODO: reduce code duplication through either variable, measurement, prior class refactoring or through additional typing. Currently, pre-commit prevents taking the union of all variable and measurement types.

    def _check_variable_dimension(
        self, var: Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES]
    ) -> None:
        """Checks that the variable is the correct dimension

        Args:
            pose_var (Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES]): The variable to check

        Raises:
            ValueError: if the variable is not the correct dimension
        """
        is_2d = isinstance(var, PoseVariable2D) or isinstance(var, LandmarkVariable2D)
        is_3d = isinstance(var, PoseVariable3D) or isinstance(var, LandmarkVariable3D)
        self._dimension_logger(is_2d, is_3d, type(var))

    def _check_measurement_dimension(
        self, measure: Union[POSE_MEASUREMENT_TYPES, POSE_LANDMARK_MEASUREMENT_TYPES]
    ) -> None:
        """Checks that the measurement is the correct dimension

        Args:
            measure (Union[POSE_MEASUREMENT_TYPES, POSE_LANDMARK_MEASUREMENT_TYPES]): The measurement to check

        Raises:
            ValueError: if the measurement is not the correct dimension
        """
        is_2d = isinstance(measure, PoseMeasurement2D) or isinstance(
            measure, PoseToLandmarkMeasurement2D
        )
        is_3d = isinstance(measure, PoseMeasurement3D) or isinstance(
            measure, PoseToLandmarkMeasurement3D
        )
        self._dimension_logger(is_2d, is_3d, type(measure))

    def _check_prior_dimension(
        self, prior_var: Union[POSE_PRIOR_TYPES, LANDMARK_PRIOR_TYPES]
    ) -> None:
        """Checks that the prior is the correct dimension

        Args:
            prior_var (Union[POSE_PRIOR_TYPES, LANDMARK_PRIOR_TYPES]): The prior to check

        Raises:
            ValueError: if the prior is not the correct dimension
        """
        is_2d = isinstance(prior_var, PosePrior2D) or isinstance(
            prior_var, LandmarkPrior2D
        )
        is_3d = isinstance(prior_var, PosePrior3D) or isinstance(
            prior_var, LandmarkPrior3D
        )
        self._dimension_logger(is_2d, is_3d, type(prior_var))
