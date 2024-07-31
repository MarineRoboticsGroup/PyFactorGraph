"""These are functions that operate on the factor graphs to create new
factor graphs from the original.

Examples:
    1) a modifier that simulates range measurements between different poses
    2) a modifier that splits a single-robot factor graph into a multi-robot one
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import copy
import itertools
import random
from attrs import define, field

from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    POSE_VARIABLE_TYPES,
    LandmarkVariable2D,
    LandmarkVariable3D,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    POSE_MEASUREMENT_TYPES,
    FGRangeMeasurement,
)
from py_factor_graph.priors import (
    LandmarkPrior2D,
    LandmarkPrior3D,
    PosePrior2D,
    PosePrior3D,
)
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.name_utils import (
    get_robot_char_from_number,
    get_time_idx_from_frame_name,
)
from py_factor_graph.utils.matrix_utils import (
    get_random_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_translation_from_transformation_matrix,
)
from py_factor_graph.utils.attrib_utils import (
    probability_validator,
    positive_float_validator,
)
from py_factor_graph.utils.logging_utils import logger


@define
class RangeMeasurementModel:
    """This class defines a sensor model for range measurements

    Args:
    """

    sensing_horizon: float = field(validator=positive_float_validator)
    range_stddev: float = field(validator=positive_float_validator)
    measurement_prob: float = field(validator=probability_validator)

    def make_measurement(
        self, association: Tuple[str, str], dist: float, timestamp: Optional[float]
    ) -> FGRangeMeasurement:
        """Makes a range measurement

        Args:
            dist (float): the distance between the two poses

        Returns:
            FGRangeMeasurement: the measurement
        """
        assert (
            0 < dist <= self.sensing_horizon
        ), f"The distance {dist} is not in the sensing horizon"
        noisy_dist = np.random.normal(dist, self.range_stddev)
        if noisy_dist < 0:
            logger.warning(f"Negative noisy distance {noisy_dist} for dist {dist}")
            noisy_dist = 0.0

        return FGRangeMeasurement(
            association=association,
            dist=noisy_dist,
            stddev=self.range_stddev,
            timestamp=timestamp,
        )


def _dist_between_variables(
    var1: Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES],
    var2: Union[POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES],
) -> float:
    """Returns the distance between two variables"""
    if isinstance(var1, PoseVariable2D) or isinstance(var1, PoseVariable3D):
        pos1 = var1.position_vector
    elif isinstance(var1, LandmarkVariable2D) or isinstance(var1, LandmarkVariable3D):
        pos1 = np.array(var1.true_position)
    else:
        raise ValueError(f"Variable {var1} not supported")

    if isinstance(var2, PoseVariable2D) or isinstance(var2, PoseVariable3D):
        pos2 = var2.position_vector
    elif isinstance(var2, LandmarkVariable2D) or isinstance(var2, LandmarkVariable3D):
        pos2 = np.array(var2.true_position)
    else:
        raise ValueError(f"Variable {var2} not supported")

    dist = np.linalg.norm(pos1 - pos2).astype(float)
    return dist


def add_landmark_at_position(
    fg: FactorGraphData,
    landmark_position: np.ndarray,
    range_measurement_model: RangeMeasurementModel,
) -> FactorGraphData:
    """Adds landmark at a random positions in the factor graph and simulates
    range measurements between the landmarks and the robot poses.

    Args:
        fg (FactorGraphData): the factor graph to modify
        num_landmarks (int): the number of landmarks to add

    Returns:
        FactorGraphData: a new factor graph with the added landmarks
    """
    new_fg = copy.deepcopy(fg)
    dim = fg.dimension
    assert landmark_position.shape == (dim, 1) or landmark_position.shape == (dim,)

    new_landmark_name = f"L{new_fg.num_landmarks}"
    new_landmark: LANDMARK_VARIABLE_TYPES
    if dim == 2:
        pos2d = (float(landmark_position[0]), float(landmark_position[1]))
        new_landmark = LandmarkVariable2D(new_landmark_name, pos2d)
    elif dim == 3:
        pos3d = (
            float(landmark_position[0]),
            float(landmark_position[1]),
            float(landmark_position[2]),
        )
        new_landmark = LandmarkVariable3D(new_landmark_name, pos3d)
    else:
        raise ValueError(f"Dimension {dim} not supported")
    new_fg.add_landmark_variable(new_landmark)

    for pose_chain in new_fg.pose_variables:
        for pose in pose_chain:
            pose_landmark_dist = _dist_between_variables(pose, new_landmark)
            if (
                pose_landmark_dist <= range_measurement_model.sensing_horizon
                and np.random.rand() < range_measurement_model.measurement_prob
            ):
                association = (pose.name, new_landmark_name)
                range_measure = range_measurement_model.make_measurement(
                    association, pose_landmark_dist, pose.timestamp
                )
                # logger.warning(f"True range {pose_landmark_dist}, noisy range {range_measure.dist}")
                new_fg.add_range_measurement(range_measure)

    return new_fg


def add_landmark_at_trajectory_center(
    fg: FactorGraphData,
    range_measurement_model: RangeMeasurementModel,
) -> FactorGraphData:
    """Adds landmark at the center of the trajectory and simulates
    range measurements between the landmarks and the robot poses.

    Args:
        fg (FactorGraphData): the factor graph to modify
        num_landmarks (int): the number of landmarks to add

    Returns:
        FactorGraphData: a new factor graph with the added landmarks
    """
    dim = fg.dimension
    assert (
        fg.num_robots == 1
    ), f"Only single robot factor graphs supported, got {fg.num_robots} robots"
    trajectory_center = np.array([0.0] * dim)

    for pose_chain in fg.pose_variables:
        for pose in pose_chain:
            trajectory_center += pose.position_vector
    trajectory_center /= fg.num_poses
    logger.info(f"Added landmark at trajectory center {trajectory_center}")
    return add_landmark_at_position(fg, trajectory_center, range_measurement_model)


def add_random_landmarks(
    fg: FactorGraphData, num_landmarks: int, range_model: RangeMeasurementModel
) -> FactorGraphData:
    """Returns the

    Args:
        fg (FactorGraphData): _description_
        num_landmarks (int): _description_

    Returns:
        FactorGraphData: _description_
    """
    new_fg = copy.deepcopy(fg)
    for _ in range(num_landmarks):
        x_min = fg.x_min
        x_max = fg.x_max
        y_min = fg.y_min
        y_max = fg.y_max
        z_min = fg.z_min
        z_max = fg.z_max
        assert x_min is not None and x_max is not None
        assert y_min is not None and y_max is not None
        assert z_min is not None and z_max is not None

        rand_x = np.random.uniform(x_min, x_max)
        rand_y = np.random.uniform(y_min, y_max)
        rand_z = np.random.uniform(z_min, z_max)

        if fg.dimension == 2:
            landmark_position = np.array([rand_x, rand_y])
        elif fg.dimension == 3:
            landmark_position = np.array([rand_x, rand_y, rand_z])
        else:
            raise ValueError(f"Dimension {fg.dimension} not supported")

        new_fg = add_landmark_at_position(new_fg, landmark_position, range_model)
    return new_fg


def remove_loop_closures(
    fg: FactorGraphData,
) -> FactorGraphData:
    """Removes loop closures from the factor graph

    Args:
        fg (FactorGraphData): the factor graph to modify

    Returns:
        FactorGraphData: a new factor graph with the loop closures removed
    """
    new_fg = copy.deepcopy(fg)
    new_fg.loop_closure_measurements.clear()
    return new_fg


def reduce_number_of_loop_closures(
    fg: FactorGraphData, percent_to_keep: float
) -> FactorGraphData:
    """Only keeps a certain percentage of the loop closures in the factor graph

    Args:
        fg (FactorGraphData): the factor graph to modify
        percent_to_keep (float): the percentage of loop closures to keep

    Returns:
        FactorGraphData: a new factor graph with the loop closures removed
    """
    new_fg = copy.deepcopy(fg)
    new_fg.loop_closure_measurements = random.sample(
        new_fg.loop_closure_measurements,
        int(len(new_fg.loop_closure_measurements) * percent_to_keep),
    )
    return new_fg


def split_single_robot_into_multi(
    fg: FactorGraphData, num_robots: int
) -> FactorGraphData:
    """Takes a single-robot factor graph and splits it into a multi-robot one.

    Args:
        fg: The single-robot factor graph.
        num_robots: The number of robots to split the factor graph into.

    Returns:
        The multi-robot factor graph.
    """
    assert (
        fg.num_robots == 1
    ), f"Expected a single-robot factor graph, but got {fg.num_robots}."

    num_total_poses = fg.num_poses
    multi_fg = FactorGraphData(fg.dimension)
    old_to_new_pose_name_mapping: Dict[str, str] = {}

    def _get_pose_chain_bound_idxs() -> List[Tuple[int, int]]:
        # set the start/stop indices for each robot using numpy
        pose_chain_indices = np.linspace(
            start=0, stop=num_total_poses, num=num_robots + 1, dtype=int
        )
        robot_pose_chain_bounds = list(
            zip(pose_chain_indices[:-1], pose_chain_indices[1:])
        )
        return robot_pose_chain_bounds

    def _copy_pose_variable_with_new_name(
        old_pose: POSE_VARIABLE_TYPES, new_name: str
    ) -> POSE_VARIABLE_TYPES:
        if isinstance(old_pose, PoseVariable2D):
            return PoseVariable2D(
                new_name,
                old_pose.true_position,
                old_pose.true_theta,
                old_pose.timestamp,
            )
        elif isinstance(old_pose, PoseVariable3D):
            return PoseVariable3D(
                new_name,
                old_pose.true_position,
                old_pose.true_rotation,
                old_pose.timestamp,
            )
        else:
            raise ValueError(f"Unknown pose type: {type(old_pose)}")

    def _add_pose_variables() -> None:
        # add the pose variables
        pose_chain = fg.pose_variables[0]
        pose_chain_bound_idxs = _get_pose_chain_bound_idxs()

        for robot_idx, (pose_chain_start, pose_chain_end) in enumerate(
            pose_chain_bound_idxs
        ):
            robot_pose_chain = pose_chain[pose_chain_start:pose_chain_end]
            robot_char = get_robot_char_from_number(robot_idx)

            for pose_idx, old_pose in enumerate(robot_pose_chain):
                pose_name = f"{robot_char}{pose_idx}"
                new_pose = _copy_pose_variable_with_new_name(old_pose, pose_name)
                old_to_new_pose_name_mapping[old_pose.name] = pose_name
                multi_fg.add_pose_variable(new_pose)

    def _copy_odom_measurement_with_new_frames(
        old_measure: POSE_MEASUREMENT_TYPES, from_frame: str, to_frame: str
    ) -> POSE_MEASUREMENT_TYPES:
        if isinstance(old_measure, PoseMeasurement2D):
            return PoseMeasurement2D(
                from_frame,
                to_frame,
                old_measure.x,
                old_measure.y,
                old_measure.theta,
                old_measure.translation_precision,
                old_measure.rotation_precision,
                old_measure.timestamp,
            )
        elif isinstance(old_measure, PoseMeasurement3D):
            return PoseMeasurement3D(
                from_frame,
                to_frame,
                old_measure.translation,
                old_measure.rotation,
                old_measure.translation_precision,
                old_measure.rotation_precision,
                old_measure.timestamp,
            )
        else:
            raise ValueError(f"Unknown pose type: {type(old_measure)}")

    def _add_odom_measurements() -> None:
        # add the odometry measurements
        pose_chain_bound_idxs = _get_pose_chain_bound_idxs()
        odom_chain = fg.odom_measurements[0]
        for robot_idx, (pose_chain_start, pose_chain_end) in enumerate(
            pose_chain_bound_idxs
        ):
            robot_odom_chain = odom_chain[pose_chain_start : pose_chain_end - 1]
            robot_char = get_robot_char_from_number(robot_idx)

            for odom_idx, odom in enumerate(robot_odom_chain):
                from_pose = f"{robot_char}{odom_idx}"
                to_pose = f"{robot_char}{odom_idx+1}"
                new_measure = _copy_odom_measurement_with_new_frames(
                    odom, from_pose, to_pose
                )
                multi_fg.add_odom_measurement(robot_idx, new_measure)

    def _add_loop_closures() -> None:
        # add the loop closures
        for loop_closure in fg.loop_closure_measurements:
            old_from_frame = loop_closure.base_pose
            old_to_frame = loop_closure.to_pose
            new_loop_closure = _copy_odom_measurement_with_new_frames(
                loop_closure,
                old_to_new_pose_name_mapping[old_from_frame],
                old_to_new_pose_name_mapping[old_to_frame],
            )
            multi_fg.add_loop_closure(new_loop_closure)

    _add_pose_variables()
    _add_odom_measurements()
    _add_loop_closures()

    assert multi_fg.num_robots == num_robots
    return multi_fg


def add_inter_robot_range_measurements(
    fg: FactorGraphData,
    range_model: RangeMeasurementModel,
) -> FactorGraphData:
    """Adds range measurements between robots within a given sensing horizon

    Args:
        fg (FactorGraphData): the original factor graph to modify
        sensing_horizon (float): the sensing horizon for the range measurements
        measurement_prob (float, optional): the probability of a measurement
            being added. Defaults to 0.3.
        range_stddev (float, optional): the standard deviation of the range

    Returns:
        FactorGraphData: a new factor graph with the added measurements
    """
    assert (
        fg.num_robots > 1
    ), "Cannot add inter-robot measurements to a single robot factor graph."
    sensing_horizon = range_model.sensing_horizon
    measurement_prob = range_model.measurement_prob
    range_stddev = range_model.range_stddev
    logger.debug(
        f"Adding inter-robot ranges currently assumes that timesteps are matched across pose chains"
    )

    new_fg = copy.deepcopy(fg)

    def _poses_have_same_timestamp(
        pose1: POSE_VARIABLE_TYPES, pose2: POSE_VARIABLE_TYPES
    ) -> bool:
        return pose1.timestamp == pose2.timestamp

    for pose_chain1, pose_chain2 in itertools.combinations(new_fg.pose_variables, 2):
        chain_1_idx, chain_2_idx = 0, 0
        chain_1_end, chain_2_end = len(pose_chain1), len(pose_chain2)

        while chain_1_idx < chain_1_end or chain_2_idx < chain_2_end:
            # use these selectors to avoid out of bounds errors
            pose1_selector = min(chain_1_idx, chain_1_end - 1)
            pose2_selector = min(chain_2_idx, chain_2_end - 1)
            pose1 = pose_chain1[pose1_selector]
            pose2 = pose_chain2[pose2_selector]

            same_timestamp = _poses_have_same_timestamp(pose1, pose2)
            one_pose_at_end = chain_1_idx == chain_1_end or chain_2_idx == chain_2_end
            if not same_timestamp and not one_pose_at_end:
                err = (
                    f"The timestamps are mismatched: {pose1.timestamp} != {pose2.timestamp}"
                    " and neither pose is at the end of its chain."
                )
                raise ValueError(err)

            dist = _dist_between_variables(pose1, pose2)

            if dist <= sensing_horizon and np.random.rand() < measurement_prob:
                association = (pose1.name, pose2.name)
                range_measure = range_model.make_measurement(
                    association, dist, range_stddev
                )
                new_fg.add_range_measurement(range_measure)

            # increment the indices
            if chain_1_idx < chain_1_end:
                chain_1_idx += 1
            if chain_2_idx < chain_2_end:
                chain_2_idx += 1

    num_range_measurements = new_fg.num_range_measurements
    assert (
        num_range_measurements > 0
    ), "No range measurements were added to the factor graph."
    return new_fg


def _add_loop_closures_based_on_existing_poses(
    new_fg: FactorGraphData, old_fg: FactorGraphData
):
    # add the loop closures
    new_poses = new_fg.pose_variables_dict
    for loop_closure in old_fg.loop_closure_measurements:
        pose1_name, pose2_name = loop_closure.base_pose, loop_closure.to_pose
        if pose1_name in new_poses and pose2_name in new_poses:
            new_loop_closure = copy.deepcopy(loop_closure)
            new_fg.add_loop_closure(new_loop_closure)


def _add_range_measurements_and_beacons_based_on_existing_poses(
    new_fg: FactorGraphData, old_fg: FactorGraphData
):
    # add the range measurements and landmarks
    new_poses = new_fg.pose_variables_dict
    old_landmarks = old_fg.landmark_variables_dict
    landmarks_to_add: Dict[str, LANDMARK_VARIABLE_TYPES] = {}
    measurements_to_add = []
    for range_meas in old_fg.range_measurements:
        pose1_name, var2_name = range_meas.association

        # need to be a little careful here to not add any range measurements to
        # unwanted poses but to add new beacon variables if necessary.
        if not new_fg.pose_exists(pose1_name):
            continue

        var2_is_pose = not var2_name.startswith("L")
        if var2_is_pose and not new_fg.pose_exists(var2_name):
            continue

        # at this point we know that we will want to add this range measurement
        measurements_to_add.append(copy.deepcopy(range_meas))

        # let's determine if we need to add a new landmark variable
        if not var2_is_pose:
            assert var2_name in old_landmarks, f"Landmark {var2_name} not found"

            # if we've already seen this landmark then do nothing
            is_new_landmark_to_add = var2_name not in landmarks_to_add
            if is_new_landmark_to_add:
                new_landmark_var = copy.deepcopy(old_landmarks[var2_name])
                landmarks_to_add[var2_name] = new_landmark_var

    # lets now sort the landmarks to add into a list, ordered by name
    sorted_landmarks_to_add = sorted(
        landmarks_to_add.values(),
        key=lambda landmark: get_time_idx_from_frame_name(landmark.name),
    )
    for landmark in sorted_landmarks_to_add:
        new_fg.add_landmark_variable(landmark)

    # now we can add the range measurements
    for range_meas in measurements_to_add:
        new_fg.add_range_measurement(range_meas)


def _add_pose_priors_based_on_existing_poses(
    new_fg: FactorGraphData, old_fg: FactorGraphData
):
    # add the pose priors
    new_poses = new_fg.pose_variables_dict
    for pose_prior in old_fg.pose_priors:
        pose_name = pose_prior.name
        if pose_name in new_poses:
            new_pose_prior = copy.deepcopy(pose_prior)
            new_fg.add_pose_prior(new_pose_prior)


def _add_landmark_priors_based_on_existing_poses(
    new_fg: FactorGraphData, old_fg: FactorGraphData
):
    # add the landmark priors
    new_landmarks = new_fg.landmark_variables_dict
    for landmark_prior in old_fg.landmark_priors:
        landmark_name = landmark_prior.name
        if landmark_name in new_landmarks:
            new_landmark_prior = copy.deepcopy(landmark_prior)
            new_fg.add_landmark_prior(new_landmark_prior)


def take_first_n_poses(fg: FactorGraphData, n: int) -> FactorGraphData:
    """Returns a factor graph with only the first n poses. Effectively tries to
    cut off at a certain timestep without strictly requiring the timesteps be
    filled.

    If 'n' is greater than the number of poses then the data of the original
    factor graph is returned (but not the exact same object)

    Args:
        fg (FactorGraphData): the factor graph to trim
        n (int): the number of poses to keep

    Returns:
        FactorGraphData: the trimmed FG
    """
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    assert isinstance(fg, FactorGraphData), "fg must be a FactorGraphData object"

    new_fg = FactorGraphData(dimension=fg.dimension)

    # add the poses and odometry measurements
    for robot_idx in range(fg.num_robots):
        for pose_var in fg.pose_variables[robot_idx][:n]:
            new_pose_var = copy.deepcopy(pose_var)
            new_fg.add_pose_variable(new_pose_var)

        # for n poses there are (n-1) odometry measurements
        for odom_meas in fg.odom_measurements[robot_idx][: n - 1]:
            new_odom_meas = copy.deepcopy(odom_meas)
            new_fg.add_odom_measurement(robot_idx, new_odom_meas)

    _add_loop_closures_based_on_existing_poses(new_fg, fg)
    _add_range_measurements_and_beacons_based_on_existing_poses(new_fg, fg)
    _add_pose_priors_based_on_existing_poses(new_fg, fg)
    _add_landmark_priors_based_on_existing_poses(new_fg, fg)

    return new_fg


def skip_first_n_poses(fg: FactorGraphData, n: int) -> FactorGraphData:
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    assert isinstance(fg, FactorGraphData), "fg must be a FactorGraphData object"

    new_fg = FactorGraphData(dimension=fg.dimension)

    # add the poses and odometry measurements
    for robot_idx in range(fg.num_robots):
        for pose_var in fg.pose_variables[robot_idx][n:]:
            new_pose_var = copy.deepcopy(pose_var)
            new_fg.add_pose_variable(new_pose_var)

        # for n poses there are (n-1) odometry measurements
        for odom_meas in fg.odom_measurements[robot_idx][n:]:
            new_odom_meas = copy.deepcopy(odom_meas)
            new_fg.add_odom_measurement(robot_idx, new_odom_meas)

    _add_loop_closures_based_on_existing_poses(new_fg, fg)
    _add_range_measurements_and_beacons_based_on_existing_poses(new_fg, fg)
    _add_pose_priors_based_on_existing_poses(new_fg, fg)
    _add_landmark_priors_based_on_existing_poses(new_fg, fg)

    return new_fg


def make_single_robot_into_multi_via_transform(
    fg: FactorGraphData, num_robots: int
) -> FactorGraphData:
    """Generates many similar trajectories (copies perturbed by a transform) from
    a single robot trajectory

    Args:
        fg (FactorGraphData): the factor graph to modify
        transform (np.ndarray): the offset to apply to the single robot trajectory

    Returns:
        FactorGraphData: a new factor graph with the modified trajectory
    """
    assert (
        fg.num_robots == 1
    ), "Cannot make a multi-robot factor graph into a single robot factor graph."
    dim = fg.dimension
    new_fg = copy.deepcopy(fg)

    def _get_pose_transform(idx: int) -> np.ndarray:
        return get_random_transformation_matrix(dim)

    def _make_new_pose(
        pose: POSE_VARIABLE_TYPES, transform: np.ndarray, new_name: str
    ) -> POSE_VARIABLE_TYPES:
        new_transform = pose.transformation_matrix @ transform
        new_translation = get_translation_from_transformation_matrix(new_transform)
        if isinstance(pose, PoseVariable2D):
            new_theta = get_theta_from_transformation_matrix(new_transform)
            pos2d = (float(new_translation[0]), float(new_translation[1]))
            return PoseVariable2D(new_name, pos2d, new_theta, pose.timestamp)
        elif isinstance(pose, PoseVariable3D):
            new_rot = get_rotation_matrix_from_transformation_matrix(new_transform)
            pos3d = (
                float(new_translation[0]),
                float(new_translation[1]),
                float(new_translation[2]),
            )
            return PoseVariable3D(new_name, pos3d, new_rot, pose.timestamp)
        else:
            raise ValueError(f"Invalid pose type: {type(pose)}")

    # make transformed poses and copy over odometry measurements
    original_pose_chain = new_fg.pose_variables[0]
    original_odom_chain = new_fg.odom_measurements[0]
    for robot_idx in range(num_robots):
        robot_char = get_robot_char_from_number(robot_idx)

        # poses
        transform = _get_pose_transform(robot_idx)
        for pose_idx, pose in enumerate(original_pose_chain):
            new_pose_name = f"{robot_char}{pose_idx}"
            new_pose = _make_new_pose(pose, transform, new_pose_name)
            new_fg.add_pose_variable(new_pose)

        # odometry
        for odom_idx, odom in enumerate(original_odom_chain):
            new_base_frame = f"{robot_char}{odom_idx}"
            new_to_frame = f"{robot_char}{odom_idx + 1}"
            if isinstance(odom, PoseMeasurement2D):
                new_odom_2d = PoseMeasurement2D(
                    new_base_frame,
                    new_to_frame,
                    odom.x,
                    odom.y,
                    odom.theta,
                    odom.translation_precision,
                    odom.rotation_precision,
                    odom.timestamp,
                )
                new_fg.add_odom_measurement(robot_idx, new_odom_2d)
            elif isinstance(odom, PoseMeasurement3D):
                new_odom_3d = PoseMeasurement3D(
                    new_base_frame,
                    new_to_frame,
                    odom.translation,
                    odom.rotation,
                    odom.translation_precision,
                    odom.rotation_precision,
                    odom.timestamp,
                )
                new_fg.add_odom_measurement(robot_idx, new_odom_3d)
            else:
                raise ValueError(f"Invalid odom type: {type(odom)}")

    return new_fg


def split_single_robot_into_multi_and_add_ranges(
    fg: FactorGraphData,
    num_robots: int,
    range_model: RangeMeasurementModel,
) -> FactorGraphData:
    """
    Splits a single robot factor graph into multiple robots and adds range
    measurements between the robots.

    Args:
        fg (FactorGraphData): the factor graph to modify
        num_robots (int): the number of robots to split the trajectory into
        range_model (RangeMeasurementModel): the range measurement model to use

    Returns:
        FactorGraphData: a new factor graph with the modified trajectory
    """
    new_fg = split_single_robot_into_multi(fg, num_robots)
    new_fg = add_inter_robot_range_measurements(new_fg, range_model)
    return new_fg


def make_beacons_into_robot_trajectory(fg: FactorGraphData) -> FactorGraphData:
    """This is useful for the HAT data, in which the beacons are the estimated
    (via high accuracy GPS or INS) trajectory of a robot. The beacon locations
    are converted into poses for a new robot and the beacon measurements are
    converted into range measurements between the poses. We will just give
    arbitrary orientation of 0 to the poses.

    Args:
        fg (FactorGraphData): the factor graph to modify

    Returns:
        FactorGraphData: a new factor graph with the modified trajectory
    """
    assert fg.dimension == 2, "Only 2D trajectories are supported."
    new_fg = copy.deepcopy(fg)
    new_robot_idx = new_fg.num_robots
    new_robot_char = get_robot_char_from_number(new_robot_idx)

    # add the new robot's poses - we set orientation to 0 to make everything easy
    for beacon_idx, beacon in enumerate(new_fg.landmark_variables):
        assert isinstance(beacon, LandmarkVariable2D)
        new_pose_name = f"{new_robot_char}{beacon_idx}"
        new_pose = PoseVariable2D(new_pose_name, beacon.true_position, 0.0)
        new_fg.add_pose_variable(new_pose)

    # add the new robot's odometry
    new_pose_chain = new_fg.pose_variables[new_robot_idx]
    num_new_poses = len(new_pose_chain)
    for odom_idx in range(num_new_poses - 1):
        base_pose = new_pose_chain[odom_idx]
        to_pose = new_pose_chain[odom_idx + 1]
        new_odom = PoseMeasurement2D(
            base_pose.name,
            to_pose.name,
            x=to_pose.true_position[0] - base_pose.true_position[0],
            y=to_pose.true_position[1] - base_pose.true_position[1],
            theta=0.0,
            translation_precision=100.0,
            rotation_precision=1000.0,
        )
        new_fg.add_odom_measurement(new_robot_idx, new_odom)

    # switch the range measurements to be between the new robot's poses
    for range_meas in new_fg.range_measurements:
        old_base_frame, old_to_frame = range_meas.association
        new_base_frame = old_base_frame
        new_to_frame = f"{new_robot_char}{int(old_to_frame[1:])}"
        range_meas.association = (new_base_frame, new_to_frame)

    # remove the beacons
    new_fg.landmark_variables = []
    new_fg.landmark_priors = []

    return new_fg


def set_all_precisions(
    fg: FactorGraphData,
    trans_precision: float,
    rot_precision: float,
    range_precision: float,
) -> FactorGraphData:
    """Sets all precisions across the factor graph to a given value per type of measurement.

    Args:
        fg (FactorGraphData): the factor graph to modify
        trans_precision (float): the translation precision to set


    Returns:
        FactorGraphData: a new factor graph with the modified precisions
    """
    new_fg = copy.deepcopy(fg)
    range_stddev = (1.0 / range_precision) ** 0.5
    for odom in new_fg.odom_measurements:
        for odom_meas in odom:
            odom_meas.translation_precision = trans_precision
            odom_meas.rotation_precision = rot_precision
    for range_meas in new_fg.range_measurements:
        range_meas.stddev = range_stddev
    return new_fg


def add_error_to_all_odom_measures(
    fg: FactorGraphData, trans_stddev: float, rot_stddev: float
) -> FactorGraphData:
    new_fg = copy.deepcopy(fg)

    for odom_chain in new_fg.odom_measurements:
        for odom in odom_chain:
            x_perturb = np.random.normal(0, trans_stddev)
            y_perturb = np.random.normal(0, trans_stddev)
            if isinstance(odom, PoseMeasurement2D):
                theta_perturb = np.random.normal(0, rot_stddev)
                odom.x = odom.x + x_perturb
                odom.y = odom.y + y_perturb
                odom.theta = odom.theta + theta_perturb
            elif isinstance(odom, PoseMeasurement3D):
                raise NotImplementedError("3D odometry not implemented")
            else:
                raise ValueError(f"Invalid odom type: {type(odom)}")

    return new_fg


def convert_to_sensor_network_localization(fg: FactorGraphData) -> FactorGraphData:
    """Generates an SNL problem by converting all poses into landmarks and
    converting all measurements into range measurements.

    Args:
        fg (FactorGraphData): the factor graph to modify

    Returns:
        FactorGraphData: the new factor graph
    """
    new_fg = FactorGraphData(dimension=fg.dimension)

    # copy all of the landmarks and their priors
    for landmark in fg.landmark_variables:
        new_landmark = copy.deepcopy(landmark)
        new_fg.add_landmark_variable(new_landmark)

    for landmark_prior in fg.landmark_priors:
        new_landmark_prior = copy.deepcopy(landmark_prior)
        new_fg.add_landmark_prior(new_landmark_prior)

    def _get_new_landmark_name() -> str:
        return f"L{new_fg.num_landmarks}"

    var_to_landmark_name_mapping = {
        landmark.name: landmark.name for landmark in fg.landmark_variables
    }

    # add all of the poses as landmarks
    for pose_chain in fg.pose_variables:
        for pose in pose_chain:
            new_landmark_name = _get_new_landmark_name()
            var_to_landmark_name_mapping[pose.name] = new_landmark_name

            if isinstance(pose, PoseVariable2D):
                new_landmark = LandmarkVariable2D(
                    new_landmark_name,
                    (pose.position_vector[0], pose.position_vector[1]),
                )
            elif isinstance(pose, PoseVariable3D):
                new_landmark = LandmarkVariable3D(
                    new_landmark_name,
                    (
                        pose.position_vector[0],
                        pose.position_vector[1],
                        pose.position_vector[2],
                    ),
                )
            else:
                raise ValueError(f"Invalid pose type: {type(pose)}")

            new_fg.add_landmark_variable(new_landmark)

    # copy pose priors as landmark priors
    for pose_prior in fg.pose_priors:
        landmark_name = var_to_landmark_name_mapping[pose_prior.name]
        if isinstance(pose_prior, PosePrior2D):
            new_landmark_prior = LandmarkPrior2D(
                landmark_name,
                (pose_prior.translation_vector[0], pose_prior.translation_vector[1]),
                pose_prior.translation_precision,
                pose_prior.timestamp,
            )
        elif isinstance(pose_prior, PosePrior3D):
            new_landmark_prior = LandmarkPrior3D(
                landmark_name,
                (
                    pose_prior.translation_vector[0],
                    pose_prior.translation_vector[1],
                    pose_prior.translation_vector[2],
                ),
                pose_prior.translation_precision,
                pose_prior.timestamp,
            )
        else:
            raise ValueError(f"Invalid pose prior type: {type(pose_prior)}")
        fg.add_landmark_prior(new_landmark_prior)

    # add all of the range measurements as range measurements
    for range_meas in fg.range_measurements:
        base_landmark, to_landmark = range_meas.association
        base_landmark = var_to_landmark_name_mapping[base_landmark]
        to_landmark = var_to_landmark_name_mapping[to_landmark]

        new_range_meas = FGRangeMeasurement(
            (base_landmark, to_landmark), range_meas.dist, range_meas.stddev
        )
        new_fg.add_range_measurement(new_range_meas)

    # add all of the odometry measurements as range measurements
    for odom_chain in fg.odom_measurements:
        for odom_measure in odom_chain:
            base_landmark = var_to_landmark_name_mapping[odom_measure.base_pose]
            to_landmark = var_to_landmark_name_mapping[odom_measure.to_pose]
            dist = np.linalg.norm(odom_measure.translation_vector).astype(float)

            # this measure of stddev is approximately right, but doesn't account
            # for "wrap around" when the translation is close to 0 (as distance
            # is non-negative)
            stddev = np.sqrt(1.0 / odom_measure.translation_precision)
            range_measure = FGRangeMeasurement(
                (base_landmark, to_landmark), dist, stddev
            )
            new_fg.add_range_measurement(range_measure)

    # add all of the loop closures as range measurements
    for loop_closure in fg.loop_closure_measurements:
        base_landmark = var_to_landmark_name_mapping[loop_closure.base_pose]
        to_landmark = var_to_landmark_name_mapping[loop_closure.to_pose]
        dist = np.linalg.norm(loop_closure.translation_vector).astype(float)
        stddev = np.sqrt(1.0 / loop_closure.translation_precision)
        range_measure = FGRangeMeasurement((base_landmark, to_landmark), dist, stddev)
        new_fg.add_range_measurement(range_measure)

    for measure in fg.pose_landmark_measurements:
        base_landmark = var_to_landmark_name_mapping[measure.pose_name]
        to_landmark = var_to_landmark_name_mapping[measure.landmark_name]
        dist = np.linalg.norm(measure.translation_vector).astype(float)
        stddev = np.sqrt(1.0 / measure.translation_precision)
        range_measure = FGRangeMeasurement((base_landmark, to_landmark), dist, stddev)
        new_fg.add_range_measurement(range_measure)

    return new_fg


def add_random_range_measurements(
    fg: FactorGraphData,
    num_measures: int,
    stddev: float,
    prob_from_pose_variable: float,
    prob_to_pose_variable: float,
) -> FactorGraphData:
    new_fg = copy.deepcopy(fg)
    # get a list of all the landmarks and poses
    pose_variables = new_fg.pose_variables_dict
    landmark_variables = new_fg.landmark_variables_dict

    # make sure that if we're expecting poses that there are pose variables
    if prob_from_pose_variable > 0.0 or prob_to_pose_variable > 0.0:
        assert len(pose_variables) > 0, "No pose variables in the factor graph"
    if prob_from_pose_variable < 1.0 or prob_to_pose_variable < 1.0:
        assert len(landmark_variables) > 0, "No landmark variables in the factor graph"

    start_num_measures = new_fg.num_range_measurements
    num_skips = 0
    existing_measures = set(new_fg.range_measures_association_dict.keys())
    while (
        new_fg.num_range_measurements < start_num_measures + num_measures
        and num_skips < 1000
    ):
        # flip a coin to decide if the measurement is between two landmarks or a pose and a landmark
        from_pose = np.random.rand() < prob_from_pose_variable
        if from_pose:
            from_name = random.choice(list(pose_variables.keys()))
        else:
            from_name = random.choice(list(landmark_variables.keys()))

        to_pose = np.random.rand() < prob_to_pose_variable
        to_name = from_name
        while to_name == from_name:
            if to_pose:
                to_name = random.choice(list(pose_variables.keys()))
            else:
                to_name = random.choice(list(landmark_variables.keys()))

        from_var = (
            pose_variables[from_name] if from_pose else landmark_variables[from_name]
        )
        to_var = pose_variables[to_name] if to_pose else landmark_variables[to_name]
        dist = _dist_between_variables(from_var, to_var)
        noisy_dist = np.random.normal(dist, stddev)
        var_association = (from_name, to_name)
        flip_association = (to_name, from_name)

        if (
            var_association in existing_measures
            or flip_association in existing_measures
        ):
            num_skips += 1
            continue

        new_fg.add_range_measurement(
            FGRangeMeasurement((from_name, to_name), noisy_dist, stddev)
        )
        existing_measures.add(var_association)

    assert (
        len(existing_measures)
        == new_fg.num_range_measurements
        == start_num_measures + num_measures
    )
    return new_fg


def make_all_ranges_perfect(fg: FactorGraphData) -> FactorGraphData:
    new_fg = copy.deepcopy(fg)
    new_fg.range_measurements = []

    # make the measured distances noiseless but keep the same stddev
    pose_vars = new_fg.pose_variables_dict
    landmark_vars = new_fg.landmark_variables_dict
    for measure in fg.range_measurements:
        from_name, to_name = measure.association
        from_var = pose_vars.get(from_name, landmark_vars.get(from_name))
        to_var = pose_vars.get(to_name, landmark_vars.get(to_name))
        assert from_var is not None, f"Variable {from_name} not found"
        assert to_var is not None, f"Variable {to_name} not found"
        dist = _dist_between_variables(from_var, to_var)
        new_fg.add_range_measurement(
            FGRangeMeasurement((from_name, to_name), dist, measure.stddev)
        )

    return new_fg


if __name__ == "__main__":
    pass
