from typing import List, Tuple, Optional, Union, overload, Dict
from attrs import define, field
import numpy as np
from scipy.stats import linregress

# from sklearn import linear_model
import matplotlib.pyplot as plt
import copy

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
    inlier_stddev_threshold: float = 3.0,
    show_outlier_rejection: bool = False,
) -> List[UncalibratedRangeMeasurement]:
    """
    We will fit a linear model to the range measurements and remove outliers. W
    """

    def _plot_inliers_and_outliers(
        inliers: List[UncalibratedRangeMeasurement],
        outliers: List[UncalibratedRangeMeasurement],
        slope: float,
        intercept: float,
    ):
        inlier_measured_dists = np.array([x.dist for x in inliers])
        inlier_true_dists = np.array([x.true_dist for x in inliers])
        outlier_measured_dists = np.array([x.dist for x in outliers])
        outlier_true_dists = np.array([x.true_dist for x in outliers])

        plt.scatter(
            inlier_measured_dists, inlier_true_dists, color="blue", label="inliers"
        )
        plt.scatter(
            outlier_measured_dists, outlier_true_dists, color="red", label="outliers"
        )
        plt.title(f"{inliers[0].association}")
        plt.legend()
        plt.xlabel("Measured distance (m)")
        plt.ylabel("True distance (m)")

        # draw the linear model up to the largest measured distance
        x = np.linspace(0, np.max(inlier_measured_dists), 100)
        y = slope * x + intercept
        plt.plot(x, y, color="black", label="linear model")

        # make sure axis is square
        plt.gca().set_aspect("equal", adjustable="box")

        plt.show(block=True)

    inliers_have_converged = False
    inlier_measurements = copy.deepcopy(uncalibrated_measurements)
    outlier_measurements = []
    while not inliers_have_converged:
        # fit a linear model to the range measurements
        linear_calibration = fit_linear_calibration_model(inlier_measurements)

        # compute the residuals and use them to find outliers
        residuals = linear_calibration.get_calibrated_residuals(inlier_measurements)
        res_stddev = np.std(residuals)
        outlier_mask = np.where(
            np.abs(residuals) > inlier_stddev_threshold * res_stddev
        )[0]

        inlier_measurements = [
            x for idx, x in enumerate(inlier_measurements) if idx not in outlier_mask
        ]
        outlier_measurements += [
            x for idx, x in enumerate(inlier_measurements) if idx in outlier_mask
        ]

        # check if we have converged
        inliers_have_converged = len(outlier_mask) == 0
        if inliers_have_converged:
            break

        # if everything is an outlier, then some nonsense is going on
        if len(outlier_mask) == len(inlier_measurements):
            logger.warning(
                f"Everything is an outlier. This is probably a bug. Returning empty list."
            )
            return []

        # remove any measurements that are outliers

    # visualize the inliers and outliers
    if show_outlier_rejection:
        if (
            "L11" in inlier_measurements[0].association
            or "L17" in inlier_measurements[0].association
        ):
            _plot_inliers_and_outliers(
                inlier_measurements,
                outlier_measurements,
                linear_calibration.slope,
                linear_calibration.intercept,
            )
        # _plot_inliers_and_outliers(
        #     inlier_measurements,
        #     outlier_measurements,
        #     linear_calibration.slope,
        #     linear_calibration.intercept,
        # )

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
            measurements, show_outlier_rejection=True
        )

    # get the calibrated measurements
    calibrated_measurements = get_linearly_calibrated_measurements(inlier_measurements)

    # update the factor graph data
    pyfg.range_measurements = calibrated_measurements

    return pyfg
