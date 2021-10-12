from typing import List
import attr
import pickle
import pathlib

from factor_graph.variables import PoseVariable, LandmarkVariable
from factor_graph.measurements import (
    PoseMeasurement,
    AmbiguousPoseMeasurement,
    FGRangeMeasurement,
    AmbiguousFGRangeMeasurement,
)
from factor_graph.priors import PosePrior, LandmarkPrior


@attr.s
class FactorGraphData:
    """
    Just a container for the data in a FactorGraph. Only considers standard
    gaussian measurements.

    pose_variables (List[PoseVariable]): a list of the pose variables
    landmark_variables (List[LandmarkVariable]): a list of the landmarks
    pose_measurements (List[List[PoseMeasurement]]): nested lists of odom measurements
    loop_closure_measurements (List[PoseMeasurement]): a list of loop closure measurements
    ambiguous_loop_closure_measurements (List[AmbiguousPoseMeasurement]): a list of ambiguous pose measurements
    range_measurements (List[FGRangeMeasurement]): a list of range measurements
    ambiguous_range_measurements (List[AmbiguousFGRangeMeasurement]): a list of ambiguous range measurements
    pose_priors (List[PosePrior]): a list of the pose priors
    landmark_priors (List[LandmarkPrior]): a list of the landmark priors
    dimension (int): the dimension of the factor graph (e.g. 2D or 3D)

    Raises:
        ValueError: inputs do not match criteria

    Returns:
        [type]: [description]
    """

    # variables
    pose_variables: List[PoseVariable] = attr.ib(factory=list)
    landmark_variables: List[LandmarkVariable] = attr.ib(factory=list)

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

    # TODO update the string
    def __str__(self):
        # TODO add ambiguous measurements
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
        line += f"odom Measurements: {len(self.pose_measurements)}\n"
        for x in self.pose_measurements:
            line += f"{x}\n"
        line += "\n"

        # add range measurements
        line += f"Range Measurements: {len(self.range_measurements)}\n"
        for x in self.range_measurements:
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

    #### Add data

    def add_pose_variable(self, pose_var: PoseVariable):
        """Adds a pose variable to the list of pose variables.

        Args:
            pose_var (PoseVariable): the pose variable to add
        """
        self.pose_variables.append(pose_var)

    def add_landmark_variable(self, landmark_var: LandmarkVariable):
        """Adds a landmark variable to the list of landmark variables.

        Args:
            landmark_var (LandmarkVariable): the landmark variable to add
        """
        self.landmark_variables.append(landmark_var)

    def add_odom_measurement(self, robot_idx: int, odom_meas: PoseMeasurement):
        """Adds an odom measurement to the list of odom measurements.

        Args:
            robot_idx (int): the index of the robot that made the measurement
            odom_meas (PoseMeasurement): the odom measurement to add
        """
        self.odom_measurements[robot_idx].append(odom_meas)

    def add_loop_closure(self, loop_closure: PoseMeasurement):
        """Adds a loop closure measurement to the list of loop closure measurements.

        Args:
            loop_closure (PoseMeasurement): the loop closure measurement to add
        """
        self.loop_closure_measurements.append(loop_closure)

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
        self.range_measurements.append(range_meas)

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

    #### saving functionalities

    def save_to_file(self, filepath: str):
        """
        Save the factor graph in the EFG format

        Args:
            filepath (str): the path of the file to write to
        """
        # TODO need to check that directory exists!
        # check is valid file type
        file_extension = pathlib.Path(filepath).suffix.strip(".")
        format_options = ["fg", "pickle"]
        assert (
            file_extension in format_options
        ), f"File extension: {file_extension} not available, must be one of {format_options}"

        if file_extension == "fg":
            self._save_to_efg_format(filepath)
        elif file_extension == "pickle":
            self._save_to_pickle_format(filepath)
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

        for pose in self.pose_variables:
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
