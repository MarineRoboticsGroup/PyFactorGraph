import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from typing import Tuple
from py_factor_graph.variables import PoseVariable, LandmarkVariable
from py_factor_graph.utils.matrix_utils import get_theta_from_rotation_matrix


def draw_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    quiver_length: float = 0.1,
    quiver_width: float = 0.01,
    color: str = "black",
) -> mpatches.FancyArrow:
    """Draws an arrow on the given axes

    Args:
        ax (plt.Axes): the axes to draw the arrow on
        x (float): the x position of the arrow
        y (float): the y position of the arrow
        theta (float): the angle of the arrow
        quiver_length (float, optional): the length of the arrow. Defaults to 0.1.
        quiver_width (float, optional): the width of the arrow. Defaults to 0.01.
        color (str, optional): color of the arrow. Defaults to "black".

    Returns:
        mpatches.FancyArrow: the arrow
    """
    dx = quiver_length * math.cos(theta)
    dy = quiver_length * math.sin(theta)
    return ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=quiver_length,
        head_length=quiver_length,
        width=quiver_width,
        color=color,
    )


def draw_line(
    ax: plt.Axes,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    color: str = "black",
) -> mlines.Line2D:
    """Draws a line on the given axes between the two points

    Args:
        ax (plt.Axes): the axes to draw the arrow on
        x_start (float): the x position of the start of the line
        y_start (float): the y position of the start of the line
        x_end (float): the x position of the end of the line
        y_end (float): the y position of the end of the line
        color (str, optional): color of the arrow. Defaults to "black".

    Returns:
        mpatches.FancyArrow: the arrow
    """
    line = mlines.Line2D([x_start, x_end], [y_start, y_end], color=color)
    ax.add_line(line)
    return line


def draw_pose_variable(ax: plt.Axes, pose: PoseVariable, color="blue"):
    true_x = pose.true_x
    true_y = pose.true_y
    true_theta = pose.true_theta
    return draw_arrow(ax, true_x, true_y, true_theta, color=color)


def draw_pose_matrix(ax: plt.Axes, pose_matrix: np.ndarray, color="blue"):
    true_x = pose_matrix[0, 2]
    true_y = pose_matrix[1, 2]
    true_theta = get_theta_from_rotation_matrix(pose_matrix[0:2, 0:2])
    return draw_arrow(ax, true_x, true_y, true_theta, color=color)


def draw_landmark_variable(ax: plt.Axes, landmark: LandmarkVariable):
    true_x = landmark.true_x
    true_y = landmark.true_y
    ax.scatter(true_x, true_y, color="green", marker=(5, 2))


def draw_loop_closure_measurement(
    ax: plt.Axes, base_loc: np.ndarray, to_pose: PoseVariable
) -> Tuple[mlines.Line2D, mpatches.FancyArrow]:
    assert base_loc.size == 2

    x_start = base_loc[0]
    y_start = base_loc[1]
    x_end = to_pose.true_x
    y_end = to_pose.true_y

    line = draw_line(ax, x_start, y_start, x_end, y_end, color="green")
    arrow = draw_pose_variable(ax, to_pose)

    return line, arrow
