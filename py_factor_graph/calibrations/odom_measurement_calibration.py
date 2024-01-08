from typing import List, Tuple, Dict
from attrs import define, field
import numpy as np
from numpy import linalg as la
from scipy.stats import linregress
import matplotlib.pyplot as plt
import copy

from py_factor_graph.measurements import PoseMeasurement2D
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.logging_utils import logger


def _validate_is_2d_pose_matrix(inst, attribute, value):
    if value.shape != (3, 3):
        raise ValueError(
            f"Expected {attribute} to be of shape (3, 3), got {value.shape}"
        )

    # check that the last row is [0, 0, 1]
    if not np.allclose(value[2, :], np.array([0, 0, 1])):
        raise ValueError(
            f"Expected {attribute} to have last row [0, 0, 1], got {value[2, :]}"
        )

    # check R.T @ R = I
    R = value[:2, :2]
    if not np.allclose(R.T @ R, np.eye(2)):
        raise ValueError(f"Expected {attribute} to have R.T @ R = I, got {R.T @ R}")


def _get_theta_from_rot_matrix(R: np.ndarray) -> float:
    """
    Get the angle of rotation from a 2D rotation matrix.
    """
    assert R.shape == (2, 2)
    theta = np.arctan2(R[1, 0], R[0, 0])
    return theta


@define
class RelPose2DResiduals:
    dx: float = field()
    dy: float = field()
    dtheta: float = field()


@define
class Uncalibrated2DRelPoseMeasurement:
    association: Tuple[str, str] = field()
    measured_rel_pose: np.ndarray = field(validator=_validate_is_2d_pose_matrix)
    true_rel_pose: np.ndarray = field(validator=_validate_is_2d_pose_matrix)
    timestamp: float = field()

    @property
    def measured_dx(self):
        return self.measured_rel_pose[0, 2]

    @property
    def measured_dy(self):
        return self.measured_rel_pose[1, 2]

    @property
    def measured_dtheta(self):
        return _get_theta_from_rot_matrix(self.measured_rel_pose[:2, :2])

    @property
    def true_dx(self):
        return self.true_rel_pose[0, 2]

    @property
    def true_dy(self):
        return self.true_rel_pose[1, 2]

    @property
    def true_dtheta(self):
        return _get_theta_from_rot_matrix(self.true_rel_pose[:2, :2])


@define
class Linear2DRelPoseCalibrationModel:
    dx_slope: float = field()
    dx_intercept: float = field()
    dy_slope: float = field()
    dy_intercept: float = field()
    dtheta_slope: float = field()
    dtheta_intercept: float = field()

    def __call__(
        self, x: List[Uncalibrated2DRelPoseMeasurement]
    ) -> List[PoseMeasurement2D]:
        assert isinstance(x, list)
        assert all([isinstance(x, Uncalibrated2DRelPoseMeasurement) for x in x])
        residuals = self.get_calibrated_residuals(x)
        residuals_arr = np.array([[x.dx, x.dy, x.dtheta] for x in residuals])
        calibrated_stddevs = np.std(residuals_arr, axis=0)
        assert calibrated_stddevs.shape == (3,)
        trans_cov = 0.5 * (calibrated_stddevs[0] ** 2 + calibrated_stddevs[1] ** 2)
        trans_precision = 1 / trans_cov
        theta_precision = 1 / calibrated_stddevs[2] ** 2
        logger.info(f"Calibrated trans stddev: {np.sqrt(trans_cov):.3f}")
        logger.info(f"Calibrated theta stddev: {calibrated_stddevs[2]:.3f}")
        calibrated_vals = self.apply_linear_calibration(x)
        calibrated_measurements = [
            PoseMeasurement2D(
                base_pose=x.association[0],
                to_pose=x.association[1],
                x=calibrated_val[0],
                y=calibrated_val[1],
                theta=calibrated_val[2],
                translation_precision=trans_precision,
                rotation_precision=theta_precision,
                timestamp=x.timestamp,
            )
            for x, calibrated_val in zip(x, calibrated_vals)
        ]
        return calibrated_measurements

    def apply_linear_calibration(
        self, uncalibrated_measurements: List[Uncalibrated2DRelPoseMeasurement]
    ) -> np.ndarray:
        measured_vals = np.array(
            [
                [x.measured_dx, x.measured_dy, x.measured_dtheta]
                for x in uncalibrated_measurements
            ]
        )
        slopes = np.array([self.dx_slope, self.dy_slope, self.dtheta_slope])
        intercepts = np.array(
            [self.dx_intercept, self.dy_intercept, self.dtheta_intercept]
        )
        predicted_vals = measured_vals * slopes + intercepts
        assert predicted_vals.shape == measured_vals.shape

        return predicted_vals

    def get_calibrated_residuals(
        self,
        uncalibrated_measurements: List[Uncalibrated2DRelPoseMeasurement],
    ) -> List[RelPose2DResiduals]:
        predicted_vals = self.apply_linear_calibration(uncalibrated_measurements)
        true_vals = np.array(
            [[x.true_dx, x.true_dy, x.true_dtheta] for x in uncalibrated_measurements]
        )

        residuals = true_vals - predicted_vals

        residual_triplets = [
            RelPose2DResiduals(
                dx=residual[0],
                dy=residual[1],
                dtheta=residual[2],
            )
            for residual in residuals
        ]
        assert len(residual_triplets) == len(uncalibrated_measurements)

        return residual_triplets


def fit_linear_calibration_model(
    uncalibrated_measurements: List[Uncalibrated2DRelPoseMeasurement],
) -> Linear2DRelPoseCalibrationModel:
    assert all(
        [
            isinstance(x, Uncalibrated2DRelPoseMeasurement)
            for x in uncalibrated_measurements
        ]
    )
    measured_dx = np.array([x.measured_dx for x in uncalibrated_measurements])
    measured_dy = np.array([x.measured_dy for x in uncalibrated_measurements])
    measured_dtheta = np.array([x.measured_dtheta for x in uncalibrated_measurements])

    true_dx = np.array([x.true_dx for x in uncalibrated_measurements])
    true_dy = np.array([x.true_dy for x in uncalibrated_measurements])
    true_dtheta = np.array([x.true_dtheta for x in uncalibrated_measurements])

    dx_slope, dx_intercept, _, _, _ = linregress(measured_dx, true_dx)
    dy_slope, dy_intercept, _, _, _ = linregress(measured_dy, true_dy)
    dtheta_slope, dtheta_intercept, _, _, _ = linregress(measured_dtheta, true_dtheta)

    calibration_model = Linear2DRelPoseCalibrationModel(
        dx_slope=dx_slope,
        dx_intercept=dx_intercept,
        dy_slope=dy_slope,
        dy_intercept=dy_intercept,
        dtheta_slope=dtheta_slope,
        dtheta_intercept=dtheta_intercept,
    )

    return calibration_model


def inspect_for_inliers_and_outliers(
    uncalibrated_measurements: List[Uncalibrated2DRelPoseMeasurement],
    inlier_stddev_threshold: float = 3.0,
) -> List[Uncalibrated2DRelPoseMeasurement]:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """
    assert all(
        [
            isinstance(x, Uncalibrated2DRelPoseMeasurement)
            for x in uncalibrated_measurements
        ]
    )
    assert len(uncalibrated_measurements) > 0
    assert inlier_stddev_threshold > 0.0

    def _plot_inliers_and_outliers(
        measured_vals: np.ndarray,
        true_vals: np.ndarray,
        outlier_mask: np.ndarray,
        slope: float,
        intercept: float,
        title: str,
    ):
        # inliers = [x for idx, x in enumerate(measurements) if idx not in outlier_mask]
        # outliers = [x for idx, x in enumerate(measurements) if idx in outlier_mask]
        inlier_measured_vals = measured_vals[~outlier_mask]
        inlier_true_vals = true_vals[~outlier_mask]
        outlier_measured_vals = measured_vals[outlier_mask]
        outlier_true_vals = true_vals[outlier_mask]

        plt.scatter(
            inlier_measured_vals, inlier_true_vals, color="blue", label="inliers"
        )
        plt.scatter(
            outlier_measured_vals, outlier_true_vals, color="red", label="outliers"
        )
        plt.title(title)
        plt.legend()
        plt.xlabel("Measured distance (m)")
        plt.ylabel("True distance (m)")

        # draw the linear model up to the largest measured distance
        x = np.linspace(0, np.max(inlier_measured_vals), 100)
        y = slope * x + intercept
        plt.plot(x, y, color="black", label="linear model")

        # make sure axis is square
        plt.gca().set_aspect("equal", adjustable="box")

        plt.show(block=True)

    inlier_measurements = copy.deepcopy(uncalibrated_measurements)
    # fit a linear model to the range measurements
    linear_calibration = fit_linear_calibration_model(inlier_measurements)

    measured_calibrated_vals = linear_calibration(inlier_measurements)
    measured_vals = np.array([[x.x, x.y, x.theta] for x in measured_calibrated_vals])
    true_vals = np.array(
        [[x.true_dx, x.true_dy, x.true_dtheta] for x in inlier_measurements]
    )

    # compute the residuals and use them to find outliers
    residuals = true_vals - measured_vals

    # compute the standard deviation of each residual independently
    res_stddev = np.std(residuals, axis=0)
    assert res_stddev.shape == (3,)

    # titles for the plots
    titles = [
        f"dx: {res_stddev[0]:.3f}",
        f"dy: {res_stddev[1]:.3f}",
        f"dtheta: {res_stddev[2]:.3f}",
    ]

    # calibration params
    calibration_slopes = [
        linear_calibration.dx_slope,
        linear_calibration.dy_slope,
        linear_calibration.dtheta_slope,
    ]

    calibration_intercepts = [
        linear_calibration.dx_intercept,
        linear_calibration.dy_intercept,
        linear_calibration.dtheta_intercept,
    ]

    for measured, true, residual, res_stddev, cl_slope, cl_intercept, title in zip(
        measured_vals.T,
        true_vals.T,
        residuals.T,
        res_stddev,
        calibration_slopes,
        calibration_intercepts,
        titles,
    ):
        # find the outliers
        outlier_mask = np.where(
            np.abs(residual) > inlier_stddev_threshold * res_stddev
        )[0]

        # visualize the inliers and outliers
        _plot_inliers_and_outliers(
            measured_vals=measured,
            true_vals=true,
            outlier_mask=outlier_mask,
            slope=cl_slope,
            intercept=cl_intercept,
            title=title,
        )

        print(
            f"Found {len(outlier_mask)} outliers out of {len(inlier_measurements)} measurements"
        )

    # if everything is an outlier, then some nonsense is going on
    if len(outlier_mask) == len(inlier_measurements):
        logger.warning(
            f"Everything is an outlier. This is probably a bug. Returning empty list."
        )
        return []

    logger.warning("Not currently rejecting outliers -- just useful for inspection")
    return inlier_measurements


def get_linearly_calibrated_measurements(
    uncalibrated_measurements: List[Uncalibrated2DRelPoseMeasurement],
) -> List[PoseMeasurement2D]:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """
    linear_calibration = fit_linear_calibration_model(uncalibrated_measurements)
    calibrated_measurements = linear_calibration(uncalibrated_measurements)
    return calibrated_measurements


def calibrate_odom_measures(
    pyfg: FactorGraphData,
) -> FactorGraphData:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """
    uncalibrated_measurements = []
    for odom_chain in pyfg.odom_measurements:
        uncalibrated_measurements.extend(odom_chain)
    # uncalibrated_measurements += pyfg.loop_closure_measurements
    true_poses = pyfg.pose_variables_dict

    # group the range measurements by association
    # e.g.,
    # - (A1, L1) and (A15, L1) will be grouped together as (A, L1)
    # - (B23, L10) and (B138, L10) will be grouped together as (B, L10)
    # - (A5, L1) and (B23, L10) will not be grouped together

    # get valid variables
    valid_variable_groups = set()
    for variable_name in true_poses.keys():
        if "L" in variable_name:
            valid_variable_groups.add(variable_name)
        else:
            valid_variable_groups.add(variable_name[0])

    # get valid associations as the cross product of valid variables
    valid_associations = set()
    for var1 in valid_variable_groups:
        for var2 in valid_variable_groups:
            # pair should be alphabetically sorted, except "L" should always be second
            if "L" in var1:
                valid_associations.add((var2, var1))
            elif "L" in var2:
                valid_associations.add((var1, var2))
            elif var1 < var2:
                valid_associations.add((var1, var2))
            else:
                valid_associations.add((var2, var1))

    def get_association_grouping(association: Tuple[str, str]) -> Tuple[str, str]:
        # assert at most one "L" in the association and must be second
        a1, a2 = association
        if "L" in a1 and "L" in a2:
            raise ValueError(f"Invalid association: {association}")
        elif "L" in a1:
            raise ValueError(f"Invalid association: {association}")

        if "L" in association[1]:
            return a1[0], a2

        a1_group = a1[0]
        a2_group = a2[0]
        if a1_group < a2_group:
            return a1_group, a2_group
        else:
            return a2_group, a1_group

    # group the range measurements by association
    association_to_measurements: Dict[
        Tuple[str, str], List[Uncalibrated2DRelPoseMeasurement]
    ] = {pair: [] for pair in valid_associations}
    for measurement in uncalibrated_measurements:
        assert isinstance(
            measurement, PoseMeasurement2D
        ), f"Expected PoseMeasurement2D, got {type(measurement)}"
        association = (measurement.base_pose, measurement.to_pose)
        association_group = get_association_grouping(association)
        if association_group not in valid_associations:
            raise ValueError(f"Invalid association: {association}")

        v1_true_pose = true_poses[measurement.base_pose].transformation_matrix
        v2_true_pose = true_poses[measurement.to_pose].transformation_matrix
        true_rel_pose = la.inv(v1_true_pose) @ v2_true_pose

        assert measurement.timestamp is not None
        uncalibrated_measurement = Uncalibrated2DRelPoseMeasurement(
            association=association,
            measured_rel_pose=measurement.transformation_matrix,
            true_rel_pose=true_rel_pose,
            timestamp=measurement.timestamp,
        )

        association_to_measurements[association_group].append(uncalibrated_measurement)

    # remove any associations that don't have any measurements
    for association, measurements in list(association_to_measurements.items()):
        if len(measurements) == 0:
            del association_to_measurements[association]

    # inspect the measurements for outliers
    for association, measurements in association_to_measurements.items():
        inspect_for_inliers_and_outliers(measurements)

    # calibrate the measurements
    # calibrated_measurements = get_linearly_calibrated_measurements(inlier_measurements)
    logger.warning("Odometry: not currently rejecting outliers or calibrating")

    # update the factor graph data
    return pyfg
