from typing import List, Dict, Tuple, Optional
from os.path import isfile
import numpy as np
import pickle
import rosbag
import pymap3d as pm
import attr

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
from py_factor_graph.utils.matrix_utils import get_rotation_matrix_from_theta
from py_factor_graph.parsing.bagmerge import merge_bag


@attr.s(frozen=True)
class LatLonAlt:
    lat = attr.ib(type=float, validator=attr.validators.instance_of(float))
    lon = attr.ib(type=float, validator=attr.validators.instance_of(float))
    alt = attr.ib(type=float, default=0.0, validator=attr.validators.instance_of(float))

    def to_enu(self, ref_latlon: "LatLonAlt") -> "ENU":
        assert isinstance(ref_latlon, LatLonAlt)

        x, y, z = pm.geodetic2enu(
            self.lat,
            self.lon,
            self.alt,
            ref_latlon.lat,
            ref_latlon.lon,
            ref_latlon.alt,
        )
        return ENU(east=x, north=y, up=z)


@attr.s(frozen=True)
class ENU:
    east = attr.ib(type=float, validator=attr.validators.instance_of(float))
    north = attr.ib(type=float, validator=attr.validators.instance_of(float))
    up = attr.ib(type=float, validator=attr.validators.instance_of(float))

    def to_latlonalt(self, ref_point: "ENU") -> "LatLonAlt":
        assert isinstance(ref_point, ENU)

        lat, lon, alt = pm.enu2geodetic(
            self.east,
            self.north,
            self.up,
            ref_point.east,
            ref_point.north,
            ref_point.up,
        )
        return LatLonAlt(lat=lat, lon=lon, alt=alt)


def get_closest_timestamp_index(timestamp: float, timestamps: List[float]) -> int:
    """
    Returns the index of the closest timestamp to the given timestamp.

    Args:
        timestamp (float): the timestamp to search for
        timestamps (List[float]): the list of timestamps to search in

    Returns:
        int: the index of the closest timestamp
    """
    assert isinstance(timestamp, float)
    assert isinstance(timestamps, list)
    assert len(timestamps) > 0
    assert all(isinstance(t, float) for t in timestamps)

    time_diffs = [abs(t - timestamp) for t in timestamps]
    return time_diffs.index(min(time_diffs))


class HATParser:
    """
    A class to parse HAT data.
    """

    def __init__(
        self,
        bag_path: str,
    ) -> None:
        """
        Initialize the HATParser.

        Args:
            bag_path: The path to the bag file.
        """
        self._bag_path: str = bag_path
        self._base_lat_lon: Optional[LatLonAlt] = None
        self._factor_graph = FactorGraphData()

        # we will store the timestamps associated with the new poses and beacons
        # so that we can look up the variable name for a given timestamp
        self._pose_times: List[float] = []
        self._pose_variable_names: List[str] = []
        self._landmark_times: List[float] = []
        self._landmark_variable_names: List[str] = []

        # variables to keep track of odometry - we will keep the velocity around
        self._most_recent_velocity: Optional[np.ndarray] = None
        self._last_diver_heading: Optional[float] = None
        self._current_diver_heading: Optional[float] = None

    @staticmethod
    def _topic_list() -> List[str]:
        expected_topics = [
            "/diver/beacon_data",
            "/diver/diver_gt",
            "/diver/diver_velocity",
            "/diver/initial_packet",
            "/diver/isam2_result",
            "/remus/remus_dr",
        ]
        return expected_topics

    def _get_pose_variable_name(self, timestamp: int) -> str:
        """
        Get the name of the pose variable for a given timestamp. If it is a new
        timestamp, then a new pose variable is created.

        Args:
            timestamp (int): The timestamp.

        Returns:
            str: The name of the pose variable.
        """
        if timestamp in self._pose_times:
            return self._pose_variable_names[self._pose_times.index(timestamp)]
        else:
            raise ValueError(f"No pose variable for timestamp {timestamp}")

    def _get_most_recent_pose_variable_name(self) -> str:
        """
        Get the name of the most recent pose variable.

        Returns:
            str: The name of the pose variable.
        """
        return self._pose_variable_names[-1]

    def _get_next_pose_variable_name(self) -> str:
        """
        Get the name of the next pose variable.

        Returns:
            str: The name of the pose variable.
        """

        num_poses = self._factor_graph.num_poses
        return f"A{num_poses}"

    def _get_beacon_variable_name(self, timestamp: int) -> str:
        """
        Get the name of the beacon variable for a given timestamp.

        Args:
            timestamp (int): The timestamp.

        Returns:
            str: The name of the beacon variable.
        """
        if timestamp in self._landmark_times:
            return self._landmark_variable_names[self._landmark_times.index(timestamp)]
        else:
            raise ValueError(f"No beacon variable for timestamp {timestamp}")

    def _get_next_beacon_variable_name(self) -> str:
        """
        Get the name of the next beacon variable.

        Returns:
            str: The name of the beacon variable.
        """
        num_beacons = self._factor_graph.num_landmarks
        return f"L{num_beacons}"

    #! there's some funky stuff in here with managing the velocity because the
    # diver swimming velocity is in the diver frame but the current velocity is
    # in the global frame. I think this works fine as long as we have an okay
    # initial estimate of the diver orientation. Otherwise, the odometry derived
    # from this will be wrong because the movement due to the flow field is not
    # in the robot frame
    def _parse_diver_velocity(self, msg) -> None:
        """
        Parse the estimated diver velocity for odometry and add it to the factor
        graph. We assume that this is the instantaneous velocity of the diver
        and the corresponding odometry measurement is from the current timestep
        to the next one.

        NOTE: this is not quite the right way to handle the uncertainties here
        but we will just roll with it for now.

        Args:
            msg (dict): The message to parse.
            factor_graph (FactorGraphData): The factor graph to add the data to.
        """

        diver_heading = msg.diver_heading
        self._last_diver_heading = self._current_diver_heading
        self._current_diver_heading = diver_heading

        diver_course_over_ground = (
            msg.diver_cog
        )  # the direction of the velocity of the diver in the world frame
        diver_speed_over_ground = msg.diver_sog  # the speed of the diver (scalar value)
        timestamp = msg.time_stamp

        diver_vel_in_world_frame = diver_speed_over_ground * np.array(
            [np.cos(diver_course_over_ground), np.sin(diver_course_over_ground)]
        )

        # transform the velocity into the diver's frame instead of the world frame
        rot_world_diver = get_rotation_matrix_from_theta(diver_heading)
        rot_diver_world = rot_world_diver.T
        diver_vel_in_diver_frame = rot_diver_world @ diver_vel_in_world_frame
        self._most_recent_velocity = diver_vel_in_diver_frame

    def _add_odom_measurement(self, next_pose_time: float):
        """This adds an odometry measurement based on the most recent pose
        variable and the next pose variable. It's a little clunky because we
        assume that the time of the next pose variable is being passed in as an
        argument, but for a quick-scripting we'll live with this!

        Args:
            next_pose_time (float): the timestamp associated with the next pose variable
        """
        last_pose_name = self._get_most_recent_pose_variable_name()
        last_pose_time = self._pose_times[-1]
        next_pose_name = self._get_next_pose_variable_name()

        assert (
            next_pose_name is not None and last_pose_name is not None
        ), f"{next_pose_name} and {last_pose_name} are None"
        time_elapsed = next_pose_time - last_pose_time

        assert self._most_recent_velocity is not None
        odom_translation_vector = time_elapsed * self._most_recent_velocity

        print(self._factor_graph.pose_variables)
        assert (
            self._current_diver_heading is not None
            and self._last_diver_heading is not None
        ), f"{self._current_diver_heading} and {self._last_diver_heading} are None"
        odom_delta_theta = self._current_diver_heading - self._last_diver_heading

        translation_variance = 5.0  # this is a lot lower than what Jesse used
        rotation_variance = 0.0436 * 2  # it seemed like this value worked for Jesse
        odom_measurement = PoseMeasurement(
            last_pose_name,
            next_pose_name,
            odom_translation_vector[0],
            odom_translation_vector[1],
            odom_delta_theta,
            translation_weight=1.0 / translation_variance,
            rotation_weight=1.0 / rotation_variance,
            timestamp=next_pose_time,
        )
        self._factor_graph.add_odom_measurement(robot_idx=0, odom_meas=odom_measurement)

    def _parse_diver_gt(self, msg) -> None:
        """
        Add diver gt data to the factor graph. Each new message will constitute
        a new PoseVariable.

        Args:
            msg (dict): The diver gt state.
        """

        # if we still don't have velocity and we already have a starting pose
        # then lets not create a new pose, as we won't have odometry chain
        if self._factor_graph.num_poses > 0:
            if self._most_recent_velocity is None:
                return
            else:
                next_pose_time = msg.diver_time_of_fix
                self._add_odom_measurement(next_pose_time)

        # get the new pose name
        new_pose_name = self._get_next_pose_variable_name()

        # get the lat/lon of the diver
        gt_diver_latlonalt = LatLonAlt(lat=msg.diver_latitude, lon=msg.diver_longitude)

        # if the base lat/lon is not set, set it
        if self._base_lat_lon is None:
            num_diver_poses = self._factor_graph.num_poses
            assert num_diver_poses == 0, "Base latlon should be set only once."
            self._base_lat_lon = gt_diver_latlonalt

        # get the local ENU coordinates of the diver relative to the first pose
        gt_diver_enu = gt_diver_latlonalt.to_enu(self._base_lat_lon)

        # make the new pose and add it to the factor graph
        pose_variable = PoseVariable(
            name=new_pose_name,
            true_position=(gt_diver_enu.east, gt_diver_enu.north),
            true_theta=msg.diver_heading,
            timestamp=msg.diver_time_of_fix,
        )
        self._factor_graph.add_pose_variable(pose_variable)

        # add the new pose name and time to the lists
        self._pose_variable_names.append(new_pose_name)
        self._pose_times.append(msg.diver_time_of_fix)

    def _parse_beacon_data(self, msg) -> None:
        """
        Adds a new landmark variable to the factor graph and adds a new range
        measurement between the new landmark variable and an existing diver pose
        variable.

        Note that all landmarks should only have a single range measurement
        added to them and should have strong priors on their position.

        Args:
            msg (): The beacon data.
        """
        measurement_timestamp = int(msg.time_of_fix)

        # construct new landmark variable and add it to the factor graph
        assert self._base_lat_lon
        beacon_latlon = LatLonAlt(lat=msg.beacon_lat, lon=msg.beacon_lon, alt=0.0)
        beacon_xyz = beacon_latlon.to_enu(self._base_lat_lon)
        beacon_id = self._get_beacon_variable_name(measurement_timestamp)
        new_landmark_variable = LandmarkVariable(
            name=beacon_id,
            true_position=(beacon_xyz.east, beacon_xyz.north),
        )
        self._factor_graph.add_landmark_variable(new_landmark_variable)

        # extract the range measurement information
        measured_range_2d = msg.two_dim_range
        beacon_loc_uncertainty = msg.beacon_loc_uncertainty
        pose_id = self._get_pose_variable_name(measurement_timestamp)

        # make sure we're adding a range measurement to an existing pose
        assert self._factor_graph.pose_exists(
            pose_id
        ), f"Pose {pose_id} does not exist in the factor graph."

        # add the range measurement to the factor graph
        new_range_measurement = FGRangeMeasurement(
            association=(beacon_id, pose_id),
            dist=measured_range_2d,
            stddev=beacon_loc_uncertainty,
            timestamp=measurement_timestamp,
        )
        self._factor_graph.add_range_measurement(new_range_measurement)

        # add a prior on the beacon location
        new_prior = LandmarkPrior(
            name=beacon_id,
            position=(beacon_xyz.east, beacon_xyz.north),
            covariance=np.eye(3) * 0.1,
        )
        self._factor_graph.add_landmark_prior(new_prior)

    def _convert_latlon_to_enu(self, lat_lon: LatLonAlt) -> ENU:
        """
        Convert a lat/lon pair to x/y.

        Args:
            lat_lon (Tuple[float, float]): The lat/lon pair.
            base_lat_lon (Tuple[float, float]): The base lat/lon pair.

        Returns:
            ENU: The x/y/z ENU location relative to the base lat/lon/alt.
        """
        assert self._base_lat_lon is not None and isinstance(
            self._base_lat_lon, LatLonAlt
        ), f"Base lat/lon is not set. {self._base_lat_lon}"
        return lat_lon.to_enu(self._base_lat_lon)

    def parse_data(self, dim=2) -> FactorGraphData:
        """
        Retrieve a pickled FactorGraphData object. Requires that the
        file ends with .pickle (e.g. "my_file.pickle").

        Args:
            filepath (str): The path to the factor graph file.
            dim (int): the dimension of the factor graph.


        Returns:
            FactorGraphData: The factor graph data.
        """
        assert isfile(self._bag_path), f"{self._bag_path} is not a file"
        assert self._bag_path.endswith(".bag"), f"{self._bag_path} is not a rosbag file"
        print(f"Parsing rosbag: {self._bag_path}")

        bag = rosbag.Bag(self._bag_path)

        expected_topics = self._topic_list()
        recorded_topics = bag.get_type_and_topic_info()[1].keys()
        assert all([topic in recorded_topics for topic in expected_topics])

        for topic, msg, timestamp in bag.read_messages(topics=expected_topics):
            if topic == "/diver/beacon_data":
                print("Parsing beacon data")
                self._parse_beacon_data(msg)
            elif topic == "/diver/diver_gt":
                print("Parsing diver gt data")
                self._parse_diver_gt(msg)
            elif topic == "/diver/diver_velocity":
                print("Parsing diver velocity data")
                self._parse_diver_velocity(msg)
            elif topic == "/diver/initial_packet":
                print("Parsing initial packet data")
                self._parse_initial_packet(msg)
            elif topic == "/remus/remus_dr":
                print("Parsing remus dr data")
                self._parse_remus_dr(msg)
            elif topic == "/diver/isam2_result":
                print("Parsing isam2 result data")
                self._parse_isam2_result(msg)
            else:
                raise ValueError(f"Unknown topic {topic}")
            print()

        return self._factor_graph


if __name__ == "__main__":
    from os.path import expanduser, join, isdir
    from os import listdir

    def merge_field_trial_data(
        field_trial_data_dir: str,
        output_filename: str,
    ) -> None:
        """
        Merge all the data from a field trial into a single file. If the output
        file already exists then nothing is done.

        Args:
            field_trial_data_dir (str): The path to the directory containing
                the field trial data.
            output_filename (str): The name of the saved file (saved in the same
            directory)
        """
        output_filepath = join(expanduser(field_trial_data_dir), output_filename)
        if isfile(output_filepath):
            return

        assert isdir(
            field_trial_data_dir
        ), f"{field_trial_data_dir} is not a valid directory"
        if "trial1" in field_trial_data_dir:
            remus_data = "BournesPond06NOV2021_1631.bag"
            diver_data = "isam_1_6NOV.bag"
        elif "trial2" in field_trial_data_dir:
            remus_data = "BournesPond06NOV2021_1652.bag"
            diver_data = "isam_2_6NOV.bag"

        remus_data_path = join(field_trial_data_dir, remus_data)
        diver_data_path = join(field_trial_data_dir, diver_data)
        assert isfile(remus_data_path), f"{remus_data_path} is not a file"
        assert isfile(diver_data_path), f"{diver_data_path} is not a file"

        # Merge the bags.
        merge_bag(remus_data_path, diver_data_path, output_filepath)
        # def merge_bag(main_bagfile, bagfile, outfile, topics=None, reindex=True):

    def _get_hat_data() -> Dict[str, List[str]]:
        data_dir = join(expanduser("~"), "data")
        hat_data_dir = join(data_dir, "hat_data")
        experiment_recordings = {}

        # the field data
        field_data_dir = join(hat_data_dir, "field_experiments")
        field_trial1_dir = join(field_data_dir, "rosbags", "trial1")
        field_trial2_dir = join(field_data_dir, "rosbags", "trial2")
        field_trial_dirs = [field_trial1_dir, field_trial2_dir]
        merged_file_name = "merged_data.bag"
        for field_trial_dir in field_trial_dirs:
            merge_field_trial_data(field_trial_dir, merged_file_name)

        field_trial_data = [
            join(field_trial_dir, merged_file_name)
            for field_trial_dir in field_trial_dirs
        ]
        assert all(isfile(f) for f in field_trial_data)
        experiment_recordings["field"] = field_trial_data

        # the sim data from Jesse's thesis
        thesis_sim_data_dir = join(
            hat_data_dir, "thesis_sims", "thesis_sims", "bagfiles"
        )
        thesis_sim_data_files = listdir(thesis_sim_data_dir)
        thesis_sim_data = [
            join(thesis_sim_data_dir, f)
            for f in thesis_sim_data_files
            if f.endswith(".bag")
        ]
        experiment_recordings["thesis_sim"] = thesis_sim_data

        # the sim data from Brendan
        brendan_sim_data_dir = join(hat_data_dir, "brendan_sims", "bagfiles")
        brendan_sim_data_files = listdir(brendan_sim_data_dir)
        brendan_sim_data = [
            join(brendan_sim_data_dir, f)
            for f in brendan_sim_data_files
            if f.endswith(".bag")
        ]
        experiment_recordings["brendan_sim"] = brendan_sim_data

        return experiment_recordings

    experiment_recordings = _get_hat_data()
    experiment = experiment_recordings["brendan_sim"][2]
    # ['/home/alan/data/hat_data/brendan_sims/bagfiles/sim_04MAY2022_NCK_Zero_Current.bag',
    # '/home/alan/data/hat_data/brendan_sims/bagfiles/sim_050522_1258_OGcode_0point1current.bag',
    # '/home/alan/data/hat_data/brendan_sims/bagfiles/sim_050622_1522_OGcode_0point3current.bag',
    # '/home/alan/data/hat_data/brendan_sims/bagfiles/sim_050622_1920_CAcode_0current.bag',
    # '/home/alan/data/hat_data/brendan_sims/bagfiles/sim_050522_1552_OGcode_0point2current.bag']

    # parse the experiment
    hat_parser = HATParser(experiment)
    pyfg = hat_parser.parse_data(experiment)

    # save the PyFG object
    save_filepath = join(
        expanduser("~"), "data", "hat_data", "brendan_sims", "pyfg_data.pickle"
    )
    pyfg.save_to_file(save_filepath)
