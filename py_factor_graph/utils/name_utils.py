import re


def get_robot_char_from_number(robot_number: int) -> str:
    """
    Get the robot character from the given robot number.
    """
    assert (
        robot_number >= 0 and robot_number < 26
    ), "Only working with single character names for now"
    char = chr(ord("A") + robot_number)
    assert char != "L", "Character L is reserved for landmarks"
    return char


def get_robot_char_from_frame_name(frame: str) -> str:
    """
    Get the robot character from the given frame.
    """
    check_is_valid_frame_name(frame)
    robot_chars = re.search(r"[a-zA-Z]+", frame).group(0)  # type: ignore
    return robot_chars


def get_robot_idx_from_char(robot_char: str) -> int:
    """
    Get the robot index from the given robot character.
    """
    return ord(robot_char) - ord("A")


def get_robot_idx_from_frame_name(frame: str) -> int:
    """
    Get the robot index from the given frame name.
    """
    check_is_valid_frame_name(frame)
    robot_char = re.search(r"[A-Z]", frame).group(0)  # type: ignore
    return get_robot_idx_from_char(robot_char)


def get_time_idx_from_frame_name(frame: str) -> int:
    """
    Get the time index from the given frame name.
    """
    check_is_valid_frame_name(frame)
    return int(re.search(r"[\d+]+", frame).group(0))  # type: ignore


def check_is_valid_frame_name(frame: str):
    """
    Runs assertions if the given frame name is valid.
    """
    assert isinstance(
        frame, str
    ), f"Frame name must be a string, not {type(frame)}: {frame}"
    assert len(re.findall(r"[a-zA-Z][\d+]+", frame)) == 1, (
        "Frame name must identify robot and pose number. "
        "E.g. A0 or B12 are both accaptable. "
        f"Received {frame}"
    )
    assert (
        len(re.findall(r"[A-Z]", frame)) == 1
    ), "Only allowing single character robot names"
