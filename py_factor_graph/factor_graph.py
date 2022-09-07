from typing import List, Dict, Set, Optional, Tuple
import attr
import pickle
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from py_factor_graph.utils.data_utils import get_theta_from_transformation_matrix

from py_factor_graph.variables import PoseVariable, LandmarkVariable
from py_factor_graph.measurements import (
    PoseMeasurement,
    AmbiguousPoseMeasurement,
    FGRangeMeasurement,
    AmbiguousFGRangeMeasurement,
)
from py_factor_graph.priors import PosePrior, LandmarkPrior
from py_factor_graph.utils.name_utils import (
    get_robot_idx_from_frame_name,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.plot_utils import (
    draw_pose_variable,
    draw_pose_matrix,
    draw_landmark_variable,
    draw_loop_closure_measurement,
    draw_line,
    draw_range_measurement,
)


@attr.s
class FactorGraphData:
    """
    Just a container for the data in a FactorGraph. Only considers standard
    gaussian measurements.

    Ambiguous measurements are used to represent cases where data association
    was uncertain

    Args:
        pose_variables (List[List[PoseVariable]]): the pose chains. Each
        different robot is a different one of the nested lists.
        landmark_variables (List[LandmarkVariable]): the landmark variables
        odom_measurements (List[List[PoseMeasurement]]): the odom measurements.
        Same structure as pose_variables.
        loop_closure_measurements (List[PoseMeasurement]): the loop closures
        ambiguous_loop_closure_measurements (List[AmbiguousPoseMeasurement]): a
        list of ambiguous loop closures.
        range_measurements (List[FGRangeMeasurement]): the range measurements
        ambiguous_range_measurements (List[AmbiguousFGRangeMeasurement]): a list
        of ambiguous range measurements.
        pose_priors (List[PosePrior]): the pose priors
        landmark_priors (List[LandmarkPrior]): the landmark priors
        dimension (int): the dimension of the factor graph (e.g. 3 for 3D)

    Raises:
        ValueError: inputs do not match criteria
    """

    # variables
    pose_variables: List[List[PoseVariable]] = attr.ib(factory=list)
    landmark_variables: List[LandmarkVariable] = attr.ib(factory=list)
    existing_pose_variables: Set[str] = attr.ib(factory=set)
    existing_landmark_variables: Set[str] = attr.ib(factory=set)

    # pose measurements
    odom_measurements: List[List[PoseMeasurement]] = attr.ib(factory=list)
    loop_closure_measurements: List[PoseMeasurement] = attr.ib(factory=list)
    ambiguous_loop_closure_measurements: List[AmbiguousPoseMeasurement] = attr.ib(
        factory=list
    )

    # range measurements
    range_measurements: List[FGRangeMeasurement] = attr.ib(factory=list)
    ambiguous_range_measurements: List[AmbiguousFGRangeMeasurement] = attr.ib(
        factory=list
    )

    # priors
    pose_priors: List[PosePrior] = attr.ib(factory=list)
    landmark_priors: List[LandmarkPrior] = attr.ib(factory=list)

    # latent dimension of the space (e.g. 2D or 3D)
    dimension: int = attr.ib(default=2)

    # useful helper values
    x_min: Optional[float] = attr.ib(default=None)
    x_max: Optional[float] = attr.ib(default=None)
    y_min: Optional[float] = attr.ib(default=None)
    y_max: Optional[float] = attr.ib(default=None)
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

    def is_empty(self) -> bool:
        """Returns whether the factor graph data is empty.

        Returns:
            bool: whether the factor graph data is empty
        """
        # if there are no pose variables, return True
        return self.num_poses == 0

    def print_summary(self) -> None:
        """Prints a summary of the factor graph data."""
        num_robots = len(self.pose_variables)
        num_poses = self.num_poses
        num_landmarks = len(self.landmark_variables)
        num_range_measurements = len(self.range_measurements)
        num_loop_closures = len(self.loop_closure_measurements)
        print(
            f"# robots: {num_robots} # poses: {num_poses} # beacons: {num_landmarks} # range measurements: {num_range_measurements} # loop closures: {num_loop_closures}"
        )

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
    def pose_variables_dict(self) -> Dict[str, PoseVariable]:
        """Returns the pose variables as a dict.

        Returns:
            Dict[str, PoseVariable]: a dict of the pose variables
        """
        pose_var_dict = {}
        for pose_chain in self.pose_variables:
            for pose in pose_chain:
                pose_var_dict[pose.name] = pose
        return pose_var_dict

    @property
    def landmark_var_dict(self) -> Dict[str, LandmarkVariable]:
        """Returns the landmark variables as a dict.

        Returns:
            Dict[str, LandmarkVariable]: a dict of the landmark variables
        """
        landmark_var_dict = {x.name: x for x in self.landmark_variables}
        return landmark_var_dict

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
        factor_vars: Set[str] = set()
        for odom_chain in self.odom_measurements:
            for odom in odom_chain:
                factor_vars.add(odom.base_pose)
                factor_vars.add(odom.to_pose)

        for range_measure in self.range_measurements:
            range_assoc = range_measure.association
            factor_vars.add(range_assoc[0])
            factor_vars.add(range_assoc[1])

        return set(self.all_variable_names) - factor_vars

    @property
    def range_measures_dict(self) -> Dict[str, List[FGRangeMeasurement]]:
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

    @property
    def loop_closure_dict(self) -> Dict[str, List[PoseMeasurement]]:
        """Returns a mapping from pose variables to their loop closure measurements.

        Returns:
            Dict[str, List[PoseMeasurement]]: the mapping from pose variables to their loop closure measurements
        """
        measures_dict: Dict[str, List[PoseMeasurement]] = {}
        for measure in self.loop_closure_measurements:
            associated_pose = measure.base_pose
            if associated_pose not in measures_dict:
                measures_dict[associated_pose] = []
            measures_dict[associated_pose].append(measure)
        return measures_dict

    def condense_odometry(self) -> "FactorGraphData":
        """Concatenates all poses that only have odometry measurements (i.e. do
        not have any range measurements or loop closures)

        Example:

        x0 -> x1 -> x2 -> x3 -> x4 -> x5
              |                 |
              l1                l2

        becomes

        x0 -> x1 ->  x4 -> x5
              |      |
              l1     l2


        Returns:
            FactorGraphData: the condensed factor graph data

        """
        condensed_data = FactorGraphData()
        range_measure_dict = self.range_measures_dict
        loop_closure_dict = self.loop_closure_dict
        old_pose_variables_dict = self.pose_variables_dict

        def _is_odometry_pose(pose: PoseVariable) -> bool:
            return (
                pose.name not in range_measure_dict
                and pose.name not in loop_closure_dict
            )

        # iterate over odometry chains
        for odom_chain in self.odom_measurements:
            for odom in odom_chain:
                # if the base pose is an odometry pose, add it to the condensed data
                if _is_odometry_pose(old_pose_variables_dict[odom.base_pose]):
                    condensed_data.add_pose_variable(
                        old_pose_variables_dict[odom.base_pose]
                    )

                # add the odometry measurement
                # condensed_data.add_odom_measurement(odom)

                # if the to pose is an odometry pose, add it to the condensed data
                if _is_odometry_pose(old_pose_variables_dict[odom.to_pose]):
                    condensed_data.add_pose_variable(
                        old_pose_variables_dict[odom.to_pose]
                    )

        raise NotImplementedError

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
                if odom.translation_weight < 1 or odom.rotation_weight < 1:
                    print(odom)
                    return False

        for range_measure in self.range_measurements:
            if range_measure.weight < 1:
                print(range_measure)
                return False

        return True

    def all_variables_have_factors(self) -> bool:
        """Checks if all variables have factors.

        Returns:
            bool: whether all variables have factors
        """
        return len(self.unconnected_variable_names) == 0

    #### Add data

    def add_pose_variable(self, pose_var: PoseVariable):
        """Adds a pose variable to the list of pose variables.

        Args:
            pose_var (PoseVariable): the pose variable to add

        Raises:
            ValueError: if the pose variable is not added in chronological order
            (time indices must be ordered to ensure that the list is always
            ordered)
        """
        robot_idx = get_robot_idx_from_frame_name(pose_var.name)
        while len(self.pose_variables) <= robot_idx:
            self.pose_variables.append([])

        # enforce that the list is sorted by time
        new_pose_time_idx = get_time_idx_from_frame_name(pose_var.name)
        if len(self.pose_variables[robot_idx]) > 0:
            last_time_idx = get_time_idx_from_frame_name(
                self.pose_variables[robot_idx][-1].name
            )
            if last_time_idx >= new_pose_time_idx:
                raise ValueError(
                    "Pose variables must be added in order of increasing time_idx"
                )

        self.pose_variables[robot_idx].append(pose_var)
        self.existing_pose_variables.add(pose_var.name)

        if self.x_min is None or self.x_min > pose_var.true_x:
            self.x_min = pose_var.true_x
        if self.x_max is None or self.x_max < pose_var.true_x:
            self.x_max = pose_var.true_x
        if self.y_min is None or self.y_min > pose_var.true_y:
            self.y_min = pose_var.true_y
        if self.y_max is None or self.y_max < pose_var.true_y:
            self.y_max = pose_var.true_y

    def add_landmark_variable(self, landmark_var: LandmarkVariable):
        """Adds a landmark variable to the list of landmark variables.

        Args:
            landmark_var (LandmarkVariable): the landmark variable to add

        Raises:
            ValueError: if the pose variable is not added in chronological order
            (time indices must be ordered to ensure that the list is always
            ordered)
        """
        if len(self.landmark_variables) > 0:
            new_landmark_idx = get_time_idx_from_frame_name(landmark_var.name)
            last_landmark_idx = get_time_idx_from_frame_name(
                self.landmark_variables[-1].name
            )
            if new_landmark_idx <= last_landmark_idx:
                print(self.landmark_variables)
                print(landmark_var)
                raise ValueError(
                    "Landmark variables must be added in order of increasing robot_idx"
                )

        self.landmark_variables.append(landmark_var)
        self.existing_landmark_variables.add(landmark_var.name)

        if self.x_min is None or self.x_min > landmark_var.true_x:
            self.x_min = landmark_var.true_x
        if self.x_max is None or self.x_max < landmark_var.true_x:
            self.x_max = landmark_var.true_x
        if self.y_min is None or self.y_min > landmark_var.true_y:
            self.y_min = landmark_var.true_y
        if self.y_max is None or self.y_max < landmark_var.true_y:
            self.y_max = landmark_var.true_y

    def add_odom_measurement(self, robot_idx: int, odom_meas: PoseMeasurement):
        """Adds an odom measurement to the list of odom measurements.

        Args:
            robot_idx (int): the index of the robot that made the measurement
            odom_meas (PoseMeasurement): the odom measurement to add
        """
        assert isinstance(odom_meas, PoseMeasurement)
        while len(self.odom_measurements) <= robot_idx:
            self.odom_measurements.append([])

        self.odom_measurements[robot_idx].append(odom_meas)

        # check that we are not adding a measurement between variables that exist
        base_pose = odom_meas.base_pose
        assert self.pose_exists(base_pose)
        to_pose = odom_meas.to_pose
        assert self.pose_exists(to_pose)

        # update max and min measurement weights
        max_odom_weight = max(odom_meas.translation_weight, odom_meas.rotation_weight)
        if self.max_measure_weight is None:
            self.max_measure_weight = max_odom_weight
        elif self.max_measure_weight < max_odom_weight:
            self.max_measure_weight = max_odom_weight

        min_odom_weight = min(odom_meas.translation_weight, odom_meas.rotation_weight)
        if self.min_measure_weight is None:
            self.min_measure_weight = min_odom_weight
        elif self.min_measure_weight > min_odom_weight:
            self.min_measure_weight = min_odom_weight

    def add_loop_closure(self, loop_closure: PoseMeasurement):
        """Adds a loop closure measurement to the list of loop closure measurements.

        Args:
            loop_closure (PoseMeasurement): the loop closure measurement to add
        """
        self.loop_closure_measurements.append(loop_closure)

        # check that we are not adding a measurement between variables that exist
        base_pose = loop_closure.base_pose
        assert self.pose_exists(base_pose)
        to_pose = loop_closure.to_pose
        assert self.pose_exists(to_pose)

        # update max and min measurement weights
        max_odom_weight = max(
            loop_closure.translation_weight, loop_closure.rotation_weight
        )
        if self.max_measure_weight is None:
            self.max_measure_weight = max_odom_weight
        elif self.max_measure_weight < max_odom_weight:
            self.max_measure_weight = max_odom_weight

        min_odom_weight = min(
            loop_closure.translation_weight, loop_closure.rotation_weight
        )
        if self.min_measure_weight is None:
            self.min_measure_weight = min_odom_weight
        elif self.min_measure_weight > min_odom_weight:
            self.min_measure_weight = min_odom_weight

    def add_ambiguous_loop_closure(self, measure: AmbiguousPoseMeasurement):
        """Adds an ambiguous loop closure measurement to the list of ambiguous loop closure measurements.

        Args:
            measure (AmbiguousPoseMeasurement): the ambiguous loop closure measurement to add
        """
        self.ambiguous_loop_closure_measurements.append(measure)

    def add_range_measurement(self, range_meas: FGRangeMeasurement):
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

    def add_ambiguous_range_measurement(self, measure: AmbiguousFGRangeMeasurement):
        """Adds an ambiguous range measurement to the list of ambiguous range measurements.

        Args:
            measure (AmbiguousFGRangeMeasurement): the ambiguous range measurement to add
        """
        self.ambiguous_range_measurements.append(measure)

    def add_pose_prior(self, pose_prior: PosePrior):
        """Adds a pose prior to the list of pose priors.

        Args:
            pose_prior (PosePrior): the pose prior to add
        """
        self.pose_priors.append(pose_prior)

    def add_landmark_prior(self, landmark_prior: LandmarkPrior):
        """Adds a landmark prior to the list of landmark priors.

        Args:
            landmark_prior (LandmarkPrior): the landmark prior to add
        """
        self.landmark_priors.append(landmark_prior)

    #### Get pose chain variable names

    def get_pose_chain_names(self):
        """Returns the pose chain variable names.

        Returns:
            List[str]: the pose chain variable names
        """
        pose_chain_names = []
        for pose_chain in self.pose_variables:
            pose_chain_names.append([pose.name for pose in pose_chain])
        return pose_chain_names

    #### saving functionalities

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
        format_options = ["fg", "pickle", "plaza"]
        assert (
            file_extension in format_options
        ), f"File extension: {file_extension} not available, must be one of {format_options}"

        if file_extension == "fg":
            self._save_to_efg_format(filepath)
        elif file_extension == "pickle":
            self._save_to_pickle_format(filepath)
        elif file_extension == "plaza":
            self._save_to_plaza_format(file_dir)
        else:
            raise ValueError(f"Unknown format: {file_extension}")

        print(f"Saved data to {filepath}")

    def _save_to_efg_format(
        self,
        data_file: str,
    ) -> None:
        """
        Save the given data to the extended factor graph format.

        Args:
            data_file (str): the path of the file to write to
        """

        def get_normal_pose_measurement_string(pose_measure: PoseMeasurement) -> str:
            """This is a utility function to get a formatted string to write to EFG
            formats for measurements which can be represented by poses (i.e.
            odometry and loop closures.

            Args:
                pose (PoseMeasurement): the measurement

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
            pose_measure: AmbiguousPoseMeasurement,
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

        def get_pose_var_string(pose: PoseVariable) -> str:
            """
            Takes a pose and returns a string in the desired format
            """
            line = "Variable Pose SE2 "

            # get local frame for pose
            pose_key = pose.name

            # add in pose information
            line += f"{pose_key} {pose.true_x:.15f} {pose.true_y:.15f} {pose.true_theta:.15f}\n"

            return line

        def get_beacon_var_string(beacon: LandmarkVariable) -> str:
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

        def get_prior_to_pin_string(prior: PosePrior) -> str:
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

        file_writer = open(data_file, "w")

        for pose_chain in self.pose_variables:
            for pose in pose_chain:
                line = get_pose_var_string(pose)
                file_writer.write(line)

        for beacon in self.landmark_variables:
            line = get_beacon_var_string(beacon)
            file_writer.write(line)

        for prior in self.pose_priors:
            line = get_prior_to_pin_string(prior)
            file_writer.write(line)

        for odom_chain in self.odom_measurements:
            for odom_measure in odom_chain:
                line = get_normal_pose_measurement_string(odom_measure)
                file_writer.write(line)

        for loop_closure in self.loop_closure_measurements:
            line = get_normal_pose_measurement_string(loop_closure)
            file_writer.write(line)

        for amb_odom_measure in self.ambiguous_loop_closure_measurements:
            line = get_ambiguous_pose_measurement_string(amb_odom_measure)
            file_writer.write(line)

        for range_measure in self.range_measurements:
            line = get_range_measurement_string(range_measure)
            file_writer.write(line)

        for amb_range_measure in self.ambiguous_range_measurements:
            line = get_ambiguous_range_measurement_string(amb_range_measure)
            file_writer.write(line)

        file_writer.close()

    def _save_to_pickle_format(self, data_file: str) -> None:
        """
        Save to efg pickle format.
        """
        pickle_file = open(data_file, "wb")
        pickle.dump(self, pickle_file)
        pickle_file.close()

    def _save_to_plaza_format(self, data_folder: str) -> None:
        """
        Save to five plaza file formats.

        Args:
            data_folder (str): the base folder to write the files to
        """
        assert (
            len(self.pose_variables) == 1
        ), ".plaza file format only supports one robot"

        def save_GT_plaza() -> None:
            """
            Save Ground Truth plaza file
            """
            filename = data_folder + "/GT.plaza"
            filewriter = open(filename, "w")

            for pose in self.pose_variables[0]:
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
                # We only take odom.x because plaza assumes we are moving in the direction of the robot's heading
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
                line = f"{range_measurement.timestamp} {range_measurement.pose_key} {range_measurement.landmark_key} {range_measurement.dist}"
                filewriter.write(line)

            filewriter.close()

        save_GT_plaza()
        save_DR_plaza()
        save_DRp_plaza()
        save_TL_plaza()
        save_TD_plaza()
        return

    def write_pose_gt_to_tum(self, data_dir: str) -> None:
        """
        Write ground truth to TUM format.

        Args:
            data_dir (str): the base folder to write the files to
        """
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print(f"Writing ground truth to TUM format in {data_dir}")
        for i, pose_chain in enumerate(self.pose_variables):
            filename = "gt_traj_" + chr(ord("A") + i) + ".tum"
            filepath = os.path.join(data_dir, filename)
            fw = open(filepath, "w")

            for pose_idx, pose in enumerate(pose_chain):
                timestamp = pose.timestamp if pose.timestamp is not None else pose_idx
                fw.write(
                    f"{timestamp} {pose.true_position[0]} {pose.true_position[1]}"
                    f" 0 0 0 {np.sin(pose.true_theta/2)} {np.cos(pose.true_theta/2)}\n"
                )

            fw.close()

    #### plotting functions ####

    def animate_groundtruth(self, pause: float = 0.01) -> None:
        """
        Animate the data.

        Args:
            pause (float): the pause time between frames
        """

        # set up plot
        fig, ax = plt.subplots(figsize=(10, 10))
        assert (
            self.x_min is not None and self.x_max is not None
        ), "x_min and x_max must be set"
        assert (
            self.y_min is not None and self.y_max is not None
        ), "y_min and y_max must be set"

        ax.set_xlim(self.x_min - 1, self.x_max + 1)
        ax.set_ylim(self.y_min - 1, self.y_max + 1)

        # these are for help visualizing the loop closures
        true_poses_dict = self.pose_variables_dict
        loop_closures = self.loop_closure_measurements
        loop_closure_dict = {
            x.base_pose: true_poses_dict[x.to_pose] for x in loop_closures
        }

        # go ahead and draw the landmarks
        for landmark in self.landmark_variables:
            draw_landmark_variable(ax, landmark)

        pose_var_plot_obj: List[mpatches.FancyArrow] = []

        cnt = 0
        num_frames_skip = 2
        max_pose_chain_length = max(
            [len(pose_chain) for pose_chain in self.pose_variables]
        )
        num_poses_show = 5

        num_full_pose_chains = 0
        for pose_chain in self.pose_variables:
            if len(pose_chain) > 0:
                num_full_pose_chains += 1

        # iterate over all the poses and visualize each pose chain at each
        # timestep
        for pose_idx in range(max_pose_chain_length):
            if cnt % num_frames_skip == 0:
                cnt = 0
            else:
                cnt += 1
                continue

            for pose_chain in self.pose_variables:

                if len(pose_chain) == 0:
                    continue

                # if past end of pose chain just grab last pose, otherwise use
                # next in chain
                if len(pose_chain) <= pose_idx:
                    pose = pose_chain[-1]
                else:
                    pose = pose_chain[pose_idx]

                # draw groundtruth solution
                var_arrow = draw_pose_variable(ax, pose)
                pose_var_plot_obj.append(var_arrow)

                # if loop closure draw it
                if pose.name in loop_closure_dict:
                    loop_line, loop_pose = draw_loop_closure_measurement(
                        ax,
                        pose.position_vector,
                        loop_closure_dict[pose.name],
                    )
                else:
                    loop_line = None
                    loop_pose = None

            plt.pause(0.001)

            # if showing loop closures let's not have them hang around forever
            if loop_line and loop_pose:
                loop_line.remove()
                loop_pose.remove()

            # we are keeping a sliding window of poses to show, this starts to
            # remove the oldest poses after a certain number of frames
            if pose_idx > num_poses_show:
                for _ in range(num_full_pose_chains):
                    pose_var_plot_obj[0].remove()
                    pose_var_plot_obj.pop(0)

        plt.close()

    def animate_odometry(self, show_gt: bool = False, pause: float = 0.01) -> None:
        """Makes an animation of the odometric chain for every robot

        Args:
            show_gt (bool, optional): whether to show the ground truth as well. Defaults to False.
            pause (float, optional): How long to pause between frames. Defaults to 0.01.
        """

        # set up plot
        fig, ax = plt.subplots(figsize=(10, 10))
        assert (
            self.x_min is not None and self.x_max is not None
        ), "x_min and x_max must be set"
        assert (
            self.y_min is not None and self.y_max is not None
        ), "y_min and y_max must be set"

        x_min = self.x_min - 0.1 * abs(self.x_min)
        x_max = self.x_max + 0.1 * abs(self.x_max)
        y_min = self.y_min - 0.1 * abs(self.y_min)
        y_max = self.y_max + 0.1 * abs(self.y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # go ahead and draw the landmarks
        for landmark in self.landmark_variables:
            draw_landmark_variable(ax, landmark)

        pose_var_plot_obj: List[mpatches.FancyArrow] = []

        odom_chain_lens = [len(x) for x in self.odom_measurements]
        pose_chain_lens = [len(x) for x in self.pose_variables]
        assert len(odom_chain_lens) == len(
            pose_chain_lens
        ), "must be same number of odometry and pose chains"
        assert all(
            x[0] + 1 == x[1] for x in zip(odom_chain_lens, pose_chain_lens)
        ), "the length of the odom chains must match with the length of the pose chains"

        num_poses_show = 5

        num_full_odom_chains = 0
        for odom_chain in self.odom_measurements:
            if len(odom_chain) > 0:
                num_full_odom_chains += 1

        # for drawing range measurements
        range_measures_dict = self.range_measures_dict
        landmark_var_dict = self.landmark_var_dict
        range_measure_objs: List[Tuple[mlines.Line2D, mpatches.Circle]] = []

        # iterate over all the poses and visualize each pose chain at each
        # timestep
        cur_poses = [
            pose_chain[0].transformation_matrix for pose_chain in self.pose_variables
        ]
        for pose_idx in range(max(pose_chain_lens)):
            for robot_idx, odom_and_pose_chain in enumerate(
                zip(self.odom_measurements, self.pose_variables)
            ):

                odom_chain, pose_chain = odom_and_pose_chain

                if len(odom_chain) == 0:
                    continue

                var_arrow = draw_pose_matrix(ax, cur_poses[robot_idx])
                pose_var_plot_obj.append(var_arrow)

                if show_gt:
                    pose_chain_idx = min(len(pose_chain) - 1, pose_idx)
                    gt_pose = pose_chain[pose_chain_idx]
                    pose_name = gt_pose.name
                    var_arrow = draw_pose_variable(ax, gt_pose, color="red")
                    pose_var_plot_obj.append(var_arrow)

                    show_ranges = True
                    if show_ranges:
                        if len(range_measure_objs) > 10:
                            line_to_remove, circle_to_remove = range_measure_objs.pop(0)
                            line_to_remove.remove()
                            circle_to_remove.remove()

                        if pose_name in range_measures_dict:
                            range_measures = range_measures_dict[pose_name]
                            for range_measure in range_measures:
                                landmark_var = landmark_var_dict[
                                    range_measure.landmark_key
                                ]
                                range_measure_objs.append(
                                    draw_range_measurement(
                                        ax, range_measure, gt_pose, landmark_var
                                    )
                                )

                ## update the odometry pose ##

                # if past end of pose chain just grab last pose, otherwise use
                # next in chain
                if len(odom_chain) <= pose_idx:
                    odom_measure = np.eye(3)
                else:
                    odom_measure = odom_chain[pose_idx].transformation_matrix

                # draw groundtruth solution
                cur_poses[robot_idx] = np.dot(cur_poses[robot_idx], odom_measure)

            plt.pause(0.001)

            if pose_idx > num_poses_show and False:
                for _ in range(num_full_odom_chains):
                    pose_var_plot_obj[0].remove()
                    pose_var_plot_obj.pop(0)

                    # if showing groundtruth need to remove that pose as well
                    if show_gt:
                        pose_var_plot_obj[0].remove()
                        pose_var_plot_obj.pop(0)

        plt.close()
