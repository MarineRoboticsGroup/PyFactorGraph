from typing import List, Tuple, Optional, Union, overload, Dict
from attrs import define, field
import numpy as np
from scipy.stats import linregress  # type: ignore
from sklearn import linear_model  # type: ignore
import matplotlib.pyplot as plt

from py_factor_graph.measurements import FGRangeMeasurement
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.logging_utils import logger


@define
class UncalibratedRangeMeasurement:
    association: Tuple[str, str] = field()
    dist: float = field()
    timestamp: float = field()
    true_dist: Optional[float] = field(default=None)

    def set_true_dist(self, true_dist: float):
        self.true_dist = true_dist


@define
class LinearCalibrationModel:
    slope: float = field()
    intercept: float = field()

    @overload
    def __call__(self, x: float) -> float:
        ...

    @overload
    def __call__(
        self, x: List[UncalibratedRangeMeasurement]
    ) -> List[FGRangeMeasurement]:
        ...

    @overload
    def __call__(
        self, x: np.ndarray[np.dtype[np.float64]]  # type: ignore
    ) -> np.ndarray[np.dtype[np.float64]]:  # type: ignore
        ...

    def __call__(
        self, x: Union[float, np.ndarray, List[UncalibratedRangeMeasurement]]
    ) -> Union[float, np.ndarray, List[FGRangeMeasurement]]:
        if isinstance(x, float):
            return self.slope * x + self.intercept
        elif isinstance(x, np.ndarray):
            return self.slope * x + self.intercept
        elif isinstance(x, list):
            assert all([isinstance(x, UncalibratedRangeMeasurement) for x in x])
            residuals = self.get_calibrated_residuals(x)
            calibrated_stddev = np.std(residuals)
            logger.info(f"Calibrated stddev: {calibrated_stddev}")
            calibrated_dists = self(np.array([x.dist for x in x]))
            calibrated_measurements = [
                FGRangeMeasurement(
                    x.association,
                    dist=calibrated_dist,
                    stddev=calibrated_stddev,
                    timestamp=x.timestamp,
                )
                for x, calibrated_dist in zip(x, calibrated_dists)
            ]
            return calibrated_measurements
        else:
            raise NotImplementedError(f"Unsupported type: {type(x)}")

    def get_calibrated_residuals(
        self,
        uncalibrated_measurements: List[UncalibratedRangeMeasurement],
    ) -> np.ndarray:
        """
        We will fit a linear model to the range measurements and remove outliers.
        """
        # make sure that all true distances are set
        assert all([x.true_dist is not None for x in uncalibrated_measurements])

        measured_distances = np.array([x.dist for x in uncalibrated_measurements])
        true_distances = np.array([x.true_dist for x in uncalibrated_measurements])
        predicted_true_distances = self(measured_distances)
        residuals = true_distances - predicted_true_distances
        return residuals


def fit_linear_calibration_model(
    uncalibrated_measurements: List[UncalibratedRangeMeasurement],
) -> LinearCalibrationModel:
    """
    We will fit a linear model to the range measurements and remove outliers.
    """
    measured_dists = np.array([x.dist for x in uncalibrated_measurements])
    true_dists = np.array([x.true_dist for x in uncalibrated_measurements])
    slope, intercept, r_value, p_value, std_err = linregress(measured_dists, true_dists)
    return LinearCalibrationModel(slope=slope, intercept=intercept)


def get_inlier_set_of_range_measurements(
    uncalibrated_measurements: List[UncalibratedRangeMeasurement],
    show_outlier_rejection: bool = False,
) -> List[UncalibratedRangeMeasurement]:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """
    if len(uncalibrated_measurements) == 0:
        return []

    if len(uncalibrated_measurements) < 5:
        logger.warning(
            f"Only {len(uncalibrated_measurements)} range measurements. Discarding"
        )
        return []

    association = uncalibrated_measurements[0].association
    if "L" in association[1]:
        data_set_name = f"Robot {association[0][0]} - Landmark {association[1]}"
    else:
        first_char = association[0][0]
        second_char = association[1][0]
        assert first_char != second_char, f"Invalid association: {association}"
        assert (
            first_char != "L" and second_char != "L"
        ), f"Invalid association: {association}"
        if first_char > second_char:
            first_char, second_char = second_char, first_char
        data_set_name = f"Range calibration: Robot {first_char} - Robot {second_char}"

    if len(uncalibrated_measurements) == 0:
        return []

    def _plot_inliers_and_outliers(
        inliers: List[UncalibratedRangeMeasurement],
        outliers: List[UncalibratedRangeMeasurement],
        ransac: linear_model.RANSACRegressor,
    ):
        inlier_measured_dists = np.array([x.dist for x in inliers])
        inlier_true_dists = np.array([x.true_dist for x in inliers])
        outlier_measured_dists = np.array([x.dist for x in outliers])
        outlier_true_dists = np.array([x.true_dist for x in outliers])

        inlier_calibrated_dists = (
            ransac.predict(inlier_measured_dists.reshape(-1, 1))
            if len(inlier_measured_dists) > 0
            else np.array([[]]).reshape(-1, 1)
        )
        outlier_calibrated_dists = (
            ransac.predict(outlier_measured_dists.reshape(-1, 1))
            if len(outlier_measured_dists) > 0
            else np.array([[]]).reshape(-1, 1)
        )

        # two subplots, on left and one on the right
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # title over both subplots
        slope = ransac.estimator_.coef_[0][0]
        intercept = ransac.estimator_.intercept_[0]
        fig.suptitle(f"{data_set_name}: slope={slope:.2f}, intercept={intercept:.2f}")

        # on the left plot show the measured vs true distances
        ax1.scatter(
            inlier_measured_dists,
            inlier_true_dists,
            color="blue",
            label="inliers",
        )
        ax1.scatter(
            outlier_measured_dists,
            outlier_true_dists,
            color="red",
            label="outliers",
        )
        all_measured_dists = np.concatenate(
            [inlier_measured_dists, outlier_measured_dists]
        )
        xmin = min(np.min(all_measured_dists), 0.0)
        xmax = np.max(all_measured_dists)
        x = np.linspace(xmin, xmax, 5)
        y = ransac.predict(x.reshape(-1, 1))
        ax1.plot(x, y, color="black", label="linear model")
        ax1.set_xlabel("Measured distance (m)")
        ax1.set_ylabel("True distance (m)")
        ax1.legend()

        # on the right plot show the calibrated vs true distances
        ax2.scatter(
            inlier_calibrated_dists,
            inlier_true_dists,
            color="blue",
            label="inliers",
        )
        ax2.scatter(
            outlier_calibrated_dists,
            outlier_true_dists,
            color="red",
            label="outliers",
        )
        ax2.set_xlabel("Calibrated distance (m)")
        ax2.set_ylabel("True distance (m)")

        # draw the linear model up to the largest measured distance
        all_calibrated_dists = np.concatenate(
            [inlier_calibrated_dists, outlier_calibrated_dists]
        )
        xmin = min(np.min(all_calibrated_dists), 0.0)
        xmax = np.max(all_calibrated_dists)
        x = np.linspace(xmin, xmax, 5)
        y = ransac.predict(x.reshape(-1, 1))
        # plt.plot(x, y, color="black", label="linear model")

        # draw a line along the y-axis to indicate the left-half plane
        all_true_dists = np.concatenate([inlier_true_dists, outlier_true_dists])
        ymin = 0.0
        ymax = np.max(all_true_dists)
        ax2.vlines(0.0, ymin, ymax, color="black", linestyle="--")

        # draw a 1-1 line
        ax2.plot(x, x, color="green", label="1-1 line")

        # make sure axis is square
        # plt.gca().set_aspect("equal", adjustable="box")
        ax1.set_aspect("equal", adjustable="box")
        ax2.set_aspect("equal", adjustable="box")

        # show labels
        plt.legend()

        plt.show(block=True)

    measured_dists = np.array([x.dist for x in uncalibrated_measurements])
    true_dists = np.array([x.true_dist for x in uncalibrated_measurements])

    # get a quick linear fit
    slope, intercept, _, _, _ = linregress(measured_dists, true_dists)
    residuals = true_dists - (slope * measured_dists + intercept)
    residuals_stddev = np.std(residuals)
    assert not np.isnan(residuals_stddev), "Residuals stddev is NaN"

    # ransac model is invalid if slope is too far from 1
    def is_model_valid(model: linear_model.LinearRegression, x, y):
        slope = model.coef_[0][0]
        return abs(slope - 1) < 0.3

    min_sample_ratio = 0.35
    ransac = linear_model.RANSACRegressor(
        residual_threshold=2 * residuals_stddev,
        min_samples=min_sample_ratio,
        is_model_valid=is_model_valid,
        max_trials=1000,
    )

    try:
        ransac.fit(measured_dists.reshape(-1, 1), true_dists.reshape(-1, 1))
    except ValueError as e:
        logger.error(
            f"{data_set_name}: Discarding all {len(uncalibrated_measurements)} measurements.\n{e}"
        )
        _plot_inliers_and_outliers([], [], ransac)
        return []

    slope = ransac.estimator_.coef_[0][0]
    if abs(slope - 1) > 0.1:
        logger.warning(
            f"{data_set_name}: {len(uncalibrated_measurements)} measurements. Calibration slope of {slope:.2f} detected. This may be due to errors in the data."
        )

    inlier_mask = ransac.inlier_mask_
    inlier_measurements = []
    outlier_measurements = []
    for measurement, is_inlier in zip(uncalibrated_measurements, inlier_mask):
        if is_inlier:
            inlier_measurements.append(measurement)
        else:
            outlier_measurements.append(measurement)

    if show_outlier_rejection:
        _plot_inliers_and_outliers(inlier_measurements, outlier_measurements, ransac)

    logger.debug(
        f"{data_set_name}: {len(inlier_measurements)} inliers, {len(outlier_measurements)} outliers"
    )

    # if len(inlier_measurements) < len(outlier_measurements) or abs(slope - 1) > 0.1:
    #     _plot_inliers_and_outliers(inlier_measurements, outlier_measurements, ransac)

    return inlier_measurements


def get_linearly_calibrated_measurements(
    uncalibrated_measurements: List[UncalibratedRangeMeasurement],
) -> List[FGRangeMeasurement]:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """
    linear_calibration = fit_linear_calibration_model(uncalibrated_measurements)
    calibrated_measurements = linear_calibration(uncalibrated_measurements)
    return calibrated_measurements


def calibrate_range_measures(
    pyfg: FactorGraphData,
    show_outlier_rejection: bool = False,
) -> FactorGraphData:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """
    uncalibrated_measurements = pyfg.range_measurements
    true_variable_positions = pyfg.variable_true_positions_dict

    # group the range measurements by association
    # e.g.,
    # - (A1, L1) and (A15, L1) will be grouped together as (A, L1)
    # - (B23, L10) and (B138, L10) will be grouped together as (B, L10)
    # - (A5, L1) and (B23, L10) will not be grouped together

    # get valid variables
    valid_variable_groups = set()
    for variable_name in true_variable_positions.keys():
        if "L" in variable_name:
            valid_variable_groups.add(variable_name)
        else:
            valid_variable_groups.add(variable_name[0])

    # get valid associations as the cross product of valid variables
    valid_associations = set()
    for var1 in valid_variable_groups:
        for var2 in valid_variable_groups:
            # pair should be alphabetically sorted, except "L" should always be second
            if var1 == var2 or "L" in var1 and "L" in var2:
                continue
            elif "L" in var1:
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
        if a1_group == a2_group:
            raise ValueError(f"Invalid measurement association: {association}")
        elif a1_group < a2_group:
            return a1_group, a2_group
        else:
            return a2_group, a1_group

    # group the range measurements by association
    association_to_measurements: Dict[
        Tuple[str, str], List[UncalibratedRangeMeasurement]
    ] = {pair: [] for pair in valid_associations}
    for measurement in uncalibrated_measurements:
        association = measurement.association
        association_group = get_association_grouping(association)
        if association_group not in valid_associations:
            raise ValueError(f"Invalid association: {association}")

        true_pos1 = np.array(true_variable_positions[association[0]])
        true_pos2 = np.array(true_variable_positions[association[1]])
        true_dist = float(np.linalg.norm(true_pos1 - true_pos2))

        assert measurement.timestamp is not None, "Timestamp must be set."

        uncalibrated_measure = UncalibratedRangeMeasurement(
            association=association,
            dist=measurement.dist,
            timestamp=measurement.timestamp,
            true_dist=true_dist,
        )
        association_to_measurements[association_group].append(uncalibrated_measure)

    # for each measurement group, get the inlier set
    inlier_measurements = []
    for association, measurements in association_to_measurements.items():
        inlier_measurements += get_inlier_set_of_range_measurements(
            measurements, show_outlier_rejection=show_outlier_rejection
        )

    # get the calibrated measurements
    calibrated_measurements = get_linearly_calibrated_measurements(inlier_measurements)

    # update the factor graph data
    pyfg.range_measurements = calibrated_measurements

    return pyfg
