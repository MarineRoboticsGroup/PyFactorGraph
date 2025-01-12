from typing import List, Tuple, Optional, Union, overload, Dict
from attrs import define, field
import numpy as np
from scipy.stats import linregress  # type: ignore
from scipy.signal import savgol_filter  # type: ignore
from sklearn import linear_model  # type: ignore
import matplotlib.pyplot as plt

from py_factor_graph.measurements import FGRangeMeasurement
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.logging_utils import logger
from py_factor_graph.variables import dist_between_variables


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


def get_range_measurements_by_association(
    pyfg: FactorGraphData,
) -> Dict[Tuple[str, str], List[FGRangeMeasurement]]:
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
    association_to_measurements: Dict[Tuple[str, str], List[FGRangeMeasurement]] = {
        pair: [] for pair in valid_associations
    }
    for measurement in uncalibrated_measurements:
        association = measurement.association
        association_group = get_association_grouping(association)
        if association_group not in valid_associations:
            raise ValueError(f"Invalid association: {association}")

        association_to_measurements[association_group].append(measurement)

    # if any association has no measurements, remove it
    association_to_measurements = {
        association: measurements
        for association, measurements in association_to_measurements.items()
        if len(measurements) > 0
    }

    # sort the measurements by timestamp
    for association, measurements in association_to_measurements.items():
        association_to_measurements[association] = sorted(
            measurements, key=lambda x: x.timestamp
        )

    return association_to_measurements


def calibrate_range_measures(
    pyfg: FactorGraphData,
    show_outlier_rejection: bool = False,
) -> FactorGraphData:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """

    measurements_by_association = get_range_measurements_by_association(pyfg)
    variables_by_name = pyfg.pose_and_landmark_variables_dict
    uncalibrated_measures_by_association: Dict[
        Tuple[str, str], List[UncalibratedRangeMeasurement]
    ] = {association: [] for association in measurements_by_association.keys()}
    for radio_association, measurements in measurements_by_association.items():
        for measure in measurements:
            assert measure.timestamp is not None, "Timestamp must be set."

            var1, var2 = (
                variables_by_name[measure.association[0]],
                variables_by_name[measure.association[1]],
            )
            assert var1 is not None, f"Variable {measure.association[0]} not found"
            assert var2 is not None, f"Variable {measure.association[1]} not found"

            uncalibrated_measures_by_association[radio_association].append(
                UncalibratedRangeMeasurement(
                    association=measure.association,
                    dist=measure.dist,
                    timestamp=measure.timestamp,
                    true_dist=dist_between_variables(var1, var2),
                )
            )

    # for each measurement group, get the inlier set
    inlier_measurements = []
    for _, uncalibrated_measurements in uncalibrated_measures_by_association.items():
        inlier_measurements += get_inlier_set_of_range_measurements(
            uncalibrated_measurements, show_outlier_rejection=show_outlier_rejection
        )

    # get the calibrated measurements
    calibrated_measurements = get_linearly_calibrated_measurements(inlier_measurements)

    # update the factor graph data
    pyfg.range_measurements = calibrated_measurements

    return pyfg


def reject_measurements_based_on_temporal_consistency(
    pyfg: FactorGraphData, show_outlier_rejection: bool = False
) -> FactorGraphData:
    """The idea here is that range measurements that are close to each other in
    time should have similar distances. If they don't, then we should discard
    them.

    Args:
        pyfg (FactorGraphData): the original data

    Returns:
        FactorGraphData: the updated data
    """
    measures_by_association = get_range_measurements_by_association(pyfg)
    inlier_measures = []
    for association, measures in measures_by_association.items():
        if len(measures) == 0:
            raise ValueError(f"No measurements for association: {association}")

        filtered_measures = apply_savgol_outlier_rejection(
            measures,
            plot_title=str(association),
            show_outlier_rejection=show_outlier_rejection,
        )
        inlier_measures += filtered_measures

    pyfg.range_measurements = inlier_measures
    return pyfg


def apply_savgol_outlier_rejection(
    original_measurements: List[FGRangeMeasurement],
    plot_title: Optional[str] = None,
    show_outlier_rejection: bool = False,
) -> List[FGRangeMeasurement]:
    """Use the Savitzky-Golay filter to smooth the data and remove outliers.

    Args:
        List[FGRangeMeasurement]: the range measurements

    Returns:
        List[FGRangeMeasurements]: the filtered range measurements
    """
    if plot_title is None:
        logger.debug(
            f"Applying Savitzky-Golay outlier rejection to {len(original_measurements)} measurements"
        )
    else:
        logger.debug(
            f"Applying Savitzky-Golay outlier rejection to {len(original_measurements)} measurements: {plot_title}"
        )

    distances = np.array([x.dist for x in original_measurements])
    timestamps_ns = np.array([x.timestamp for x in original_measurements])

    # convert timestamps to seconds and subtract the first timestamp
    timestamps = (timestamps_ns - timestamps_ns[0]) / 1e9

    # get the update frequency on the data
    update_freq_hz = np.median(np.diff(timestamps))
    window_size_seconds = 2.0
    window_size_samples = int(window_size_seconds / update_freq_hz)

    poly_degree = 2
    if window_size_samples <= poly_degree:
        logger.warning(
            f"Window size of {window_size_samples} samples is too small for polynomial degree of {poly_degree}. Rejecting all measurements."
        )
        return []

    smoothed_distances = savgol_filter(distances, window_size_samples, poly_degree)

    # Calculate residuals (difference between original and smoothed values)
    residuals = np.abs(distances - smoothed_distances)

    # Compute the threshold based on the median and MAD of residuals
    mad_residuals = np.median(np.abs(residuals - np.median(residuals)))
    median_abs_deviation_threshold = 5.0
    threshold_value = median_abs_deviation_threshold * mad_residuals

    # Detect outliers
    outliers = residuals > threshold_value
    inliers = ~outliers

    # plot the data, with outliers in red. Size of the point is 1
    if show_outlier_rejection:
        plt.scatter(timestamps[inliers], distances[inliers], label="Inliers", s=1)
        plt.scatter(
            timestamps[outliers],
            distances[outliers],
            label="Outliers",
            s=1,
            color="red",
        )

        # draw the smoothed data
        plt.plot(
            timestamps,
            smoothed_distances,
            label="Smoothed Distances",
            color="orange",
            linewidth=1.0,
            linestyle="--",
        )

        plt.xlabel("Timestamp")
        plt.ylabel("Distance")
        if plot_title is not None:
            plt.title(plot_title)
        plt.legend()
        plt.show()

    # return the inliers, with the smoothed distances in place of the original
    smoothed_inlier_ranges = [
        FGRangeMeasurement(
            x.association,
            dist=smoothed_dist,
            stddev=x.stddev,
            timestamp=x.timestamp,
        )
        for x, smoothed_dist, is_inlier in zip(
            original_measurements, smoothed_distances, inliers
        )
        if is_inlier
    ]
    return smoothed_inlier_ranges

    # return [x for x, is_inlier in zip(original_measurements, inliers) if is_inlier]
