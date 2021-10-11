from typing import Tuple, List, Optional
import numpy as np
import attr
import os
import pickle

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

    Args:
        pose_variables (List[PoseVariable]): a list of the pose variables
        landmark_variables (List[LandmarkVariable]): a list of the landmarks
        pose_measurements (List[PoseMeasurement]): a list of odom
            measurements
        ambiguous_pose_measurements (List[AmbiguousPoseMeasurement]): a list of
            ambiguous pose measurements
        range_measurements (List[FGRangeMeasurement]): a list of range
            measurements
        ambiguous_range_measurements (List[AmbiguousFGRangeMeasurement]): a list
            of ambiguous range measurements
        pose_priors (List[PosePrior]): a list of the pose priors
        landmark_priors (List[LandmarkPrior]): a list of the landmark priors
    """

    pose_variables: List[PoseVariable] = attr.ib(factory=list)
    landmark_variables: List[LandmarkVariable] = attr.ib(factory=list)
    pose_measurements: List[PoseMeasurement] = attr.ib(factory=list)
    ambiguous_pose_measurements: List[AmbiguousPoseMeasurement] = attr.ib(factory=list)
    range_measurements: List[FGRangeMeasurement] = attr.ib(factory=list)
    ambiguous_range_measurements: List[AmbiguousFGRangeMeasurement] = attr.ib(
        factory=list
    )
    pose_priors: List[PosePrior] = attr.ib(factory=list)
    landmark_priors: List[LandmarkPrior] = attr.ib(factory=list)
    _dimension: int = attr.ib(default=2)

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
        line += f"Dimension: {self._dimension}\n\n"
        return line

    @property
    def num_poses(self):
        return len(self.pose_variables)

    @property
    def true_values_vector(self) -> np.ndarray:
        """
        returns the true values in a vectorized form
        """
        vect: List[float] = []

        # add in pose translations
        for pose in self.pose_variables:
            x, y = pose.true_position
            vect.append(x)
            vect.append(y)

        # add in landmark positions
        for pos in self.landmark_variables:
            x, y = pos.true_position
            vect.append(x)
            vect.append(y)

        # add in rotation measurements
        for pose in self.pose_variables:
            rot = pose.rotation_matrix.T.flatten()
            for v in rot:
                vect.append(v)

        # add in distance variables
        for range_measurement in self.range_measurements:
            vect.append(range_measurement.dist)

        vect.append(1.0)
        return np.array(vect)

    @property
    def num_translations(self):
        return self.num_poses + self.num_landmarks

    @property
    def num_landmarks(self):
        return len(self.landmark_variables)

    @property
    def dimension(self):
        return self._dimension

    @property
    def poses_and_landmarks_dimension(self):
        d = self.dimension

        # size due to translations
        n_trans = self.num_translations
        mat_dim = n_trans * d

        # size due to rotations
        n_pose = self.num_poses
        mat_dim += n_pose * d * d

        return mat_dim

    @property
    def distance_variables_dimension(self):
        mat_dim = self.num_range_measurements + 1
        return mat_dim

    @property
    def num_range_measurements(self):
        return len(self.range_measurements)

    @property
    def num_pose_measurements(self):
        return len(self.pose_measurements)

    @property
    def num_total_measurements(self):
        return self.num_range_measurements + self.num_pose_measurements

    @property
    def dist_measurements_vect(self) -> np.ndarray:
        """
        Get a vector of the distance measurements

        Returns:
            np.ndarray: a vector of the distance measurements
        """
        return np.array([meas.dist for meas in self.range_measurements])

    @property
    def weighted_dist_measurements_vect(self) -> np.ndarray:
        """
        Get of the distance measurements weighted by their precision
        """
        return self.dist_measurements_vect * self.measurements_weight_vect

    @property
    def measurements_weight_vect(self) -> np.ndarray:
        """
        Get the weights of the measurements
        """
        return np.array([meas.weight for meas in self.range_measurements])

    @property
    def sum_weighted_measurements_squared(self) -> float:
        """
        Get the sum of the squared weighted measurements
        """
        weighted_dist_vect = self.weighted_dist_measurements_vect
        dist_vect = self.dist_measurements_vect
        return np.dot(weighted_dist_vect, dist_vect)

    #### Add data

    def add_pose_variable(self, pose_var: PoseVariable):
        self.pose_variables.append(pose_var)

    def add_landmark_variable(self, landmark_var: LandmarkVariable):
        self.landmark_variables.append(landmark_var)

    def add_pose_measurement(self, odom_meas: PoseMeasurement):
        self.pose_measurements.append(odom_meas)

    def add_ambiguous_pose_measurement(self, measure: AmbiguousPoseMeasurement):
        self.ambiguous_pose_measurements.append(measure)

    def add_range_measurement(self, range_meas: FGRangeMeasurement):
        self.range_measurements.append(range_meas)

    def add_ambiguous_range_measurement(self, measure: AmbiguousFGRangeMeasurement):
        self.ambiguous_range_measurements.append(measure)

    def add_pose_prior(self, pose_prior: PosePrior):
        self.pose_priors.append(pose_prior)

    def add_landmark_prior(self, landmark_prior: LandmarkPrior):
        self.landmark_priors.append(landmark_prior)

    #### Accessors for the data

    def get_range_measurement_pose(self, measure: FGRangeMeasurement) -> PoseVariable:
        """Gets the pose associated with the range measurement

        Arguments:
            measure (FGRangeMeasurement): the range measurement

        Returns:
            PoseVariable: the pose variable associated with the range
                measurement
        """
        pose_name = measure.association[0]
        pose_idx = int(pose_name[1:])
        return self.pose_variables[pose_idx]

    def get_range_measurement_landmark(
        self, measure: FGRangeMeasurement
    ) -> LandmarkVariable:
        """Returns the landmark variable associated with this range measurement

        Arguments:
            measure (FGRangeMeasurement): the range measurement

        Returns:
            LandmarkVariable: the landmark variable associated with the range
                measurement
        """
        landmark_name = measure.association[1]
        landmark_idx = int(landmark_name[1:])
        return self.landmark_variables[landmark_idx]

    def get_pose_translation_variable_indices(
        self, pose: PoseVariable
    ) -> Tuple[int, int]:
        """
        Get the indices [start, stop) for the translation variable corresponding to this pose
        in the factor graph

        Args:
            pose (PoseVariable): the pose variable

        Returns:
            Tuple[int, int]: [start, stop) the start and stop indices
        """
        assert pose in self.pose_variables
        pose_idx = self.pose_variables.index(pose)
        d = self.dimension

        # get the start and stop indices for the translation variables
        start = pose_idx * d
        stop = (pose_idx + 1) * d

        return (start, stop)

    def get_landmark_translation_variable_indices(
        self, landmark: LandmarkVariable
    ) -> Tuple[int, int]:
        """
        Get the indices [start, stop) for the translation variable corresponding to this landmark
        in the factor graph

        Args:
            landmark (LandmarkVariable): the landmark variable

        Returns:
            Tuple[int, int]: [start, stop) the start and stop indices
        """
        assert landmark in self.landmark_variables
        landmark_idx = self.landmark_variables.index(landmark)

        # offset due to the pose translations
        d = self.dimension
        offset = self.num_poses * d

        # get the start and stop indices for the translation variables
        start = landmark_idx * d + offset
        stop = start + d

        return (start, stop)

    def get_pose_rotation_variable_indices(self, pose: PoseVariable) -> Tuple[int, int]:
        """
        Get the indices [start, stop) for the rotation variable corresponding to
        this pose in the factor graph

        Args:
            pose (PoseVariable): the pose variable

        Returns:
            Tuple[int, int]: [start, stop) the start and stop indices
        """
        assert pose in self.pose_variables
        pose_idx = self.pose_variables.index(pose)
        d = self.dimension

        # need an offset to skip over all the translation variables
        offset = self.num_translations * d

        # get the start and stop indices
        start = (pose_idx * d * d) + offset
        stop = start + d * d

        return (start, stop)

    def get_range_dist_variable_indices(self, measurement: FGRangeMeasurement) -> int:
        """
        Get the index for the distance variable corresponding to
        this measurement in the factor graph

        Args:
            measurement (FGRangeMeasurement): the measurement

        Returns:
            int: the index of the distance variable
        """
        assert measurement in self.range_measurements
        measure_idx = self.range_measurements.index(measurement)
        d = self.dimension

        # need an offset to skip over all the translation and rotation
        # variables
        range_offset = self.num_translations * d
        range_offset += self.num_poses * d * d

        # get the start and stop indices
        range_idx = (measure_idx) + range_offset

        return range_idx

    def save_to_file(self, data_dir: str, format: str):
        """
        Save the factor graph in the EFG format

        Args:
            file_name (str): the name of the file to write to
        """

        format_options = ["efg", "pickle"]
        assert (
            format in format_options
        ), f"Save format: {format} not available, must be one of {format_options}"

        if format == "efg":
            filepath = os.path.join(data_dir, "factor_graph.fg")
            self._save_to_efg_format(filepath)
        elif format == "pickle":
            filepath = os.path.join(data_dir, "factor_graph.pkl")
            self._save_to_pickle_format(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved data to {filepath}")

    def _save_to_efg_format(
        self,
        data_file: str,
    ) -> None:
        """
        Save the given data to the extended factor graph format.
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

        for odom_measure in self.pose_measurements:
            line = get_normal_pose_measurement_string(odom_measure)
            file_writer.write(line)

        for amb_odom_measure in self.ambiguous_pose_measurements:
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
