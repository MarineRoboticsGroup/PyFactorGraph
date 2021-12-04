from typing import List, Dict, Set, Optional
import attr
import pickle
import pathlib
import os
import numpy as np
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

    def pose_exists(self, pose_var_name: str) -> bool:
        """Returns whether pose variables exist.

        Args:
            pose_var_name (str): the name of the pose variable

        Returns:
            bool: whether pose variables exist
        """
        return len(self.pose_variables) > 0

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
        while len(self.odom_measurements) <= robot_idx:
            self.odom_measurements.append([])

        self.odom_measurements[robot_idx].append(odom_meas)

        # check that we are not adding a measurement between variables that exist
        base_pose = odom_meas.base_pose
        assert base_pose in self.existing_pose_variables
        to_pose = odom_meas.to_pose
        assert to_pose in self.existing_pose_variables

    def add_loop_closure(self, loop_closure: PoseMeasurement):
        """Adds a loop closure measurement to the list of loop closure measurements.

        Args:
            loop_closure (PoseMeasurement): the loop closure measurement to add
        """
        self.loop_closure_measurements.append(loop_closure)

        # check that we are not adding a measurement between variables that exist
        base_pose = loop_closure.base_pose
        assert base_pose in self.existing_pose_variables
        to_pose = loop_closure.to_pose
        assert to_pose in self.existing_pose_variables

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
        assert (
            var1 in self.existing_pose_variables
            or var1 in self.existing_landmark_variables
        )
        assert (
            var2 in self.existing_pose_variables
            or var2 in self.existing_landmark_variables
        )
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

            for pose in pose_chain:
                fw.write(
                    f"{pose.timestamp} {pose.true_position[0]} {pose.true_position[1]}"
                    f" 0 0 0 {np.sin(pose.true_theta/2)} {np.cos(pose.true_theta/2)}\n"
                )

            fw.close()
