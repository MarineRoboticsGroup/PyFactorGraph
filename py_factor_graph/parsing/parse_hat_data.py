# type: ignore
from typing import List, Dict, Tuple, Optional, Any
from queue import Queue
from os.path import isfile, join, isdir, dirname
from os import listdir, remove, makedirs
from pathlib import Path
import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.serde import deserialize_cdr, ros1_to_cdr, serialize_cdr, cdr_to_ros1
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import std_msgs__msg__String as String
import pymap3d as pm
import attr

from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    FGRangeMeasurement,
)
from py_factor_graph.priors import LandmarkPrior
from py_factor_graph.factor_graph import (
    FactorGraphData,
)
from py_factor_graph.utils.matrix_utils import (
    get_rotation_matrix_from_theta,
    get_measurement_precisions_from_covariances,
)
from py_factor_graph.parsing.bagmerge import merge_bags
from py_factor_graph.modifiers import make_beacons_into_robot_trajectory


#### Ugly steps to enable the use of custom messages for the rosbags package ####
hat_msg_dir: Path = Path("/home/alan/ros_hat/msg")


def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix("")
    if "msg" not in name.parts:
        name = name.parent / "msg" / name.name
    return str(name)


def is_ros_msg_file(path: Path) -> bool:
    """Check if file is a ros message file."""
    return path.suffix == ".msg" and "msg" in path.parts


assert isdir(hat_msg_dir), f"{hat_msg_dir} is not a directory"
add_types = {}

msg_paths = [p for p in hat_msg_dir.glob("**/*.msg") if is_ros_msg_file(p)]
for path_str in msg_paths:
    msg_path = Path(path_str)
    msg_def = msg_path.read_text(encoding="utf-8")
    add_types.update(get_types_from_msg(msg_def, guess_msgtype(msg_path)))

register_types(add_types)
from rosbags.typesys.types import ros_hat__msg__BeaconData as BeaconData
from rosbags.typesys.types import ros_hat__msg__DiverVelocity as DiverVelocity
from rosbags.typesys.types import ros_hat__msg__DiverGPS as DiverGPS
from rosbags.typesys.types import ros_hat__msg__InitialPacket as InitialPacket

#### Ugly steps to enable the use of custom messages for the rosbags package ####


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


def convert_global_degrees_to_enu_degrees(compass_heading_degrees: float) -> float:
    """
    Converts a compass heading to a cartesian angle.

    Args:
        compass_heading: The compass heading.

    Returns:
        (float): The cartesian angle.
    """
    return (360.0 - compass_heading_degrees + 90.0) % 360.0


def convert_global_degrees_to_enu_radians(compass_heading_degrees: float) -> float:
    """
    Converts a compass heading to a cartesian angle.

    Args:
        compass_heading: The compass heading.

    Returns:
        (float): The cartesian angle.
    """
    enu_degrees = convert_global_degrees_to_enu_degrees(compass_heading_degrees)
    enu_radians = np.deg2rad(enu_degrees)
    return enu_radians


class HATParser:
    """
    A class to parse HAT data.
    """

    def __init__(
        self,
        bag_path: str,
        dimension: int = 2,
    ) -> None:
        """
        Initialize the HATParser.

        Args:
            bag_path: The path to the bag file.
        """
        self._bag_path: str = bag_path
        self._base_lat_lon: Optional[LatLonAlt] = None
        self._factor_graph = FactorGraphData(dimension=dimension)

        self._beacon_data_msgs: List[BeaconData] = []
        self._diver_gps_msgs: List[DiverGPS] = []
        self._gps_times_vector = None
        self._diver_velocity_msgs: List[DiverVelocity] = []
        self._initial_packet_msg: Optional[Tuple[int, InitialPacket]] = None

        self._pose_timestamps = []
        self._pose_names = []

    @property
    def _TOPIC_LIST(
        self,
    ) -> List[str]:
        expected_topics = [
            "/diver/beacon_data",
            "/diver/diver_gps",
            "/diver/diver_velocity",
            "/diver/initial_packet",
        ]
        return expected_topics

    def _get_next_pose_variable_name(self) -> str:
        """
        Get the name of the next pose variable.

        Returns:
            str: The name of the pose variable.
        """

        num_poses = self._factor_graph.num_poses
        return f"A{num_poses}"

    def _get_next_beacon_variable_name(self) -> str:
        """
        Get the name of the next beacon variable.

        Returns:
            str: The name of the beacon variable.
        """
        num_beacons = self._factor_graph.num_landmarks
        return f"L{num_beacons}"

    @staticmethod
    def _get_diver_velocity_in_diver_frame(msg) -> np.ndarray:
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
        #! there's some funky stuff in here with managing the velocity because the
        # diver swimming velocity is in the diver frame but the current velocity is
        # in the global frame. I think this works fine as long as we have an okay
        # initial estimate of the diver orientation. Otherwise, the odometry derived
        # from this will be wrong because the movement due to the flow field is not
        # in the robot frame

        # convert all directions from geodesic to ENU
        diver_heading_enu_radians = convert_global_degrees_to_enu_radians(
            msg.diver_heading
        )
        diver_course_over_ground_enu_radians = convert_global_degrees_to_enu_radians(
            msg.diver_cog
        )
        diver_velocity_direction_wrt_heading = (
            diver_course_over_ground_enu_radians - diver_heading_enu_radians
        )
        diver_speed_over_ground = msg.diver_sog  # the speed of the diver (scalar value)

        diver_speed_over_ground = 0.25
        diver_vel_in_diver_frame = diver_speed_over_ground * np.array(
            [
                np.cos(diver_velocity_direction_wrt_heading),
                np.sin(diver_velocity_direction_wrt_heading),
            ]
        )
        return diver_vel_in_diver_frame

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

    def _update_msg_lists(self, topic, msg, timestamp):
        if topic == "/diver/beacon_data":
            self._beacon_data_msgs.append(msg)
        elif topic == "/diver/diver_gps":
            self._diver_gps_msgs.append(msg)
        elif topic == "/diver/diver_velocity":
            self._diver_velocity_msgs.append(msg)
        elif topic == "/diver/initial_packet":
            assert (
                self._initial_packet_msg is None
            ), "Initial packet should only be received once."

            # convert timestamp to seconds from nanoseconds
            timestamp /= 1e9
            time_msg_pair = (timestamp, msg)
            self._initial_packet_msg = time_msg_pair
        else:
            raise ValueError(f"Unknown topic {topic}")

    def _remove_messages_before_initial_packet(self):
        """
        Remove all messages before the initial packet.
        """
        assert self._initial_packet_msg is not None, "Initial packet is not set."
        initial_packet_time = self._initial_packet_msg[0]
        self._beacon_data_msgs = [
            msg
            for msg in self._beacon_data_msgs
            if msg.time_of_fix >= initial_packet_time
        ]
        self._diver_gps_msgs = [
            msg
            for msg in self._diver_gps_msgs
            if msg.diver_time_of_fix >= initial_packet_time
        ]
        self._diver_velocity_msgs = [
            msg
            for msg in self._diver_velocity_msgs
            if msg.time_stamp >= initial_packet_time
        ]

    def set_base_latlonalt(self):
        """
        Set the base lat/lon/alt of the system.
        """
        assert self._initial_packet_msg is not None, "Initial packet is not set."

        # set the lat/lon/alt of the starting position
        msg = self._initial_packet_msg[1]
        diver_depth = msg.diver_depth
        diver_latitude = msg.diver_latitude
        diver_longitude = msg.diver_longitude
        diver_latlonalt = LatLonAlt(
            lat=diver_latitude, lon=diver_longitude, alt=diver_depth
        )
        self._base_lat_lon = diver_latlonalt

    def _set_gps_times_vector(self):
        """
        Set the gps times vector.
        """
        self._gps_times_vector = np.array(
            [msg.diver_time_of_fix for msg in self._diver_gps_msgs]
        )

    def _get_nearest_gps_fix(self, time_stamp):
        """
        Get the nearest gps fix to the given time stamp.
        """
        assert self._gps_times_vector is not None, "Gps times vector is not set."
        gps_fix_diff = np.abs(self._gps_times_vector - time_stamp)
        gps_fix_idx = np.argmin(gps_fix_diff)
        return self._diver_gps_msgs[gps_fix_idx]

    def _add_base_pose_for_vel_msg(self, msg) -> str:
        """Adds the base pose corresponding to this velocity message (because the velocity is being used to get odometry)

        Args:
            msg (DiverVelocity): the estimated velocity of the diver

        Returns:
            str: the name of the base pose
        """
        # add the base pose for this velocity message
        base_pose_name = self._get_next_pose_variable_name()
        base_pose_gps_fix = self._get_nearest_gps_fix(msg.time_stamp)
        base_pose_xyz = self._convert_latlon_to_enu(
            LatLonAlt(
                lat=base_pose_gps_fix.diver_latitude,
                lon=base_pose_gps_fix.diver_longitude,
                alt=self._base_lat_lon.alt,  #! not using the true depth
            )
        )
        cur_msg_heading_enu_rad = convert_global_degrees_to_enu_radians(
            msg.diver_heading
        )
        base_pose = PoseVariable2D(
            name=base_pose_name,
            true_position=(base_pose_xyz.east, base_pose_xyz.north),
            true_theta=cur_msg_heading_enu_rad,
            timestamp=base_pose_gps_fix.diver_time_of_fix,
        )
        self._pose_timestamps.append(base_pose.timestamp)
        self._pose_names.append(base_pose_name)
        self._factor_graph.add_pose_variable(base_pose)
        return base_pose_name

    def _find_pose_nearest_to_timestamp(self, timestamp) -> str:
        """
        Find the pose variable name closest to the given timestamp
        """
        nearest_pose_idx = np.argmin(np.abs(self._pose_timestamps_arr - timestamp))
        return self._pose_names[nearest_pose_idx]

    def _compose_odometry_from_diver_velocity(self):
        """Iterates over all of the diver velocity messages and adds the
        corresponding odometry measurements and pose variables to the factor
        graph"""
        for vel_msg_idx, msg in enumerate(self._diver_velocity_msgs[:-1]):

            if vel_msg_idx == 0:
                base_pose_name = self._add_base_pose_for_vel_msg(msg)
            next_msg = self._diver_velocity_msgs[vel_msg_idx + 1]
            next_pose_name = self._add_base_pose_for_vel_msg(next_msg)

            # get the change in heading
            cur_msg_heading_enu_rad = convert_global_degrees_to_enu_radians(
                msg.diver_heading
            )
            next_msg_heading_enu_rad = convert_global_degrees_to_enu_radians(
                next_msg.diver_heading
            )
            delta_heading = next_msg_heading_enu_rad - cur_msg_heading_enu_rad

            # get the change in position
            delta_time = next_msg.time_stamp - msg.time_stamp
            diver_vel_in_diver_frame = self._get_diver_velocity_in_diver_frame(msg)
            diver_delta_xy = diver_vel_in_diver_frame * delta_time

            # make the odometry message
            translation_variance = 5.0  # this is a lot lower than what Jesse used
            rotation_variance = 0.0436 * 2  # it seemed like this value worked for Jesse
            (
                trans_precision,
                rot_precision,
            ) = get_measurement_precisions_from_covariances(
                translation_variance, rotation_variance
            )
            odometry = PoseMeasurement2D(
                base_pose=base_pose_name,
                to_pose=next_pose_name,
                x=diver_delta_xy[0],
                y=diver_delta_xy[1],
                theta=delta_heading,
                translation_precision=trans_precision,
                rotation_precision=rot_precision,
                timestamp=float(msg.time_stamp),
            )

            # add the odometry measurement to the factor graph
            robot_idx = 0
            self._factor_graph.add_odom_measurement(robot_idx, odometry)
            base_pose_name = next_pose_name

    def _add_range_measurements_from_beacon_data(self):
        """
        Add range measurements from beacon data.
        """
        for msg in self._beacon_data_msgs:
            # get the beacon variable
            beacon_name = self._get_next_beacon_variable_name()
            beacon_latlon = LatLonAlt(
                lat=msg.beacon_lat,
                lon=msg.beacon_lon,
                alt=self._base_lat_lon.alt,  #! not using the true depth
            )
            beacon_xy = self._convert_latlon_to_enu(beacon_latlon)
            beacon_position = (beacon_xy.east, beacon_xy.north)
            beacon_var = LandmarkVariable2D(
                name=beacon_name, true_position=beacon_position
            )
            self._factor_graph.add_landmark_variable(beacon_var)

            # apply a prior on the beacon position
            prior_precision = 1.0 / msg.beacon_loc_uncertainty
            beacon_prior = LandmarkPrior(beacon_name, beacon_position, prior_precision)
            self._factor_graph.add_landmark_prior(beacon_prior)

            # add the range measurement
            pose_name = self._find_pose_nearest_to_timestamp(msg.time_of_fix)
            dist_measured = msg.two_dim_range
            data_association = (pose_name, beacon_name)
            range_measure = FGRangeMeasurement(
                association=data_association,
                dist=dist_measured,
                stddev=2.00,  #! this is a hardcoded value
                timestamp=msg.time_of_fix,
            )
            self._factor_graph.add_range_measurement(range_measure)

    def _print_msg_list_status(self):
        # print number of messages in each list
        print(f"{len(self._beacon_data_msgs)} beacon data messages")
        print(f"{len(self._diver_gps_msgs)} diver gps messages")
        print(f"{len(self._diver_velocity_msgs)} diver velocity messages")

    def _fill_factor_graph_from_msg_lists(self):

        # remove all messages that were received before the initial packet
        self._remove_messages_before_initial_packet()
        self._print_msg_list_status()

        # set vector of when the gps fixes were received
        self._set_gps_times_vector()

        # set the initial state from the initial packet
        self.set_base_latlonalt()

        # add all the diver pose measurements
        self._compose_odometry_from_diver_velocity()

        # add all the beacon measurements
        self._pose_timestamps_arr = np.array(self._pose_timestamps)
        self._add_range_measurements_from_beacon_data()

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

        with Reader(self._bag_path) as reader:
            expected_topics = self._TOPIC_LIST
            connections = [x for x in reader.connections if x.topic in expected_topics]
            for connection, timestamp, raw_data in reader.messages(
                connections=connections
            ):
                topic = connection.topic
                if topic not in expected_topics:
                    continue
                msgtype = connection.msgtype
                msg = deserialize_cdr(ros1_to_cdr(raw_data, msgtype), msgtype)
                self._update_msg_lists(topic, msg, timestamp)

        self._fill_factor_graph_from_msg_lists()
        return self._factor_graph

    def add_terminate_to_end_of_rosbag(self, output_bag_path: str) -> None:
        """Adds a terminate signal to the end of the rosbag

        Args:
            output_bag_path (str): The path to the output rosbag
        """

        if isfile(output_bag_path):
            print(f"Removing {output_bag_path}")
            remove(output_bag_path)

        terminate_topic = "/diver/HNT_trigger"
        terminate_signal = "terminate"
        terminate_msg_type = String.__msgtype__
        with Reader(self._bag_path) as reader, Writer(output_bag_path) as writer:
            conn_map = {}
            print(f"Num reader connections: {len(reader.connections)}")
            for conn in reader.connections:
                if conn.topic not in conn_map:
                    conn_map[conn.topic] = writer.add_connection(
                        topic=conn.topic,
                        msgtype=conn.msgtype,
                        msgdef=conn.msgdef,
                        md5sum=conn.md5sum,
                    )

            last_timestamp = None
            for conn, timestamp, raw_data in reader.messages():
                writer.write(conn_map[conn.topic], timestamp, raw_data)
                last_timestamp = timestamp

            assert isinstance(last_timestamp, int)
            terminate_connection = writer.add_connection(
                terminate_topic, terminate_msg_type
            )

            terminate_data = serialize_cdr(String(terminate_signal), terminate_msg_type)
            writer.write(
                terminate_connection,
                timestamp=last_timestamp + 1,
                data=terminate_data,
            )
        print(f"Added terminate signal to {output_bag_path}")

    def remove_repeat_initial_packets(self, output_bag_path: str) -> None:
        if isfile(output_bag_path):
            print(f"Removing {output_bag_path}")
            remove(output_bag_path)

        with Reader(self._bag_path) as reader, Writer(output_bag_path) as writer:
            conn_map = {}
            print(f"Num reader connections: {len(reader.connections)}")
            for conn in reader.connections:
                if conn.topic not in conn_map:
                    conn_map[conn.topic] = writer.add_connection(
                        topic=conn.topic,
                        msgtype=conn.msgtype,
                        msgdef=conn.msgdef,
                        md5sum=conn.md5sum,
                    )

            has_written_initial_packet = False
            for conn, timestamp, raw_data in reader.messages():
                if (
                    conn.topic == "/diver/initial_packet"
                    and not has_written_initial_packet
                ):
                    has_written_initial_packet = True
                    writer.write(conn_map[conn.topic], timestamp, raw_data)
                else:
                    writer.write(conn_map[conn.topic], timestamp, raw_data)


def _get_aug_18_experiments() -> List[str]:
    data_dir = "/home/alan/data/hat_data/18AUG2022"
    files = [
        "2022-08-18-09-43-18_run1_kayak.bag",
        "2022-08-18-10-26-34_run2_kayak.bag",
        "2022-08-18-11-09-26_run3_kayak.bag",
        "2022-08-18-12-09-32_run4_kayak.bag",
        "2022-08-18-13-08-35_run5_kayak.bag",
    ]
    return [join(data_dir, f) for f in files]


if __name__ == "__main__":

    # parse the experiment
    experiment = "/home/alan/data/hat_data/16OCT2022/2022-10-16-12-14-46_terminate_added_cleaned.bag"

    for experiment in _get_aug_18_experiments():
        hat_parser = HATParser(experiment)
        pyfg = hat_parser.parse_data()
        save_filepath = experiment.replace(".bag", "_pyfg.pickle")
        pyfg.save_to_file(save_filepath)
