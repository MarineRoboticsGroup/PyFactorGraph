from typing import List, Tuple, Optional, Union, overload
from attrs import define, field
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import copy

from py_factor_graph.measurements import FGRangeMeasurement
import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)


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
        self, x: np.ndarray[np.dtype[np.float64]]
    ) -> np.ndarray[np.dtype[np.float64]]:
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
        measurements: List[UncalibratedRangeMeasurement],
        outlier_mask: np.ndarray,
    ):
        inliers = [x for idx, x in enumerate(measurements) if idx not in outlier_mask]
        outliers = [x for idx, x in enumerate(measurements) if idx in outlier_mask]
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
        plt.legend()
        plt.show(block=True)

    inliers_have_converged = False
    inlier_measurements = copy.deepcopy(uncalibrated_measurements)
    while not inliers_have_converged:
        # fit a linear model to the range measurements
        linear_calibration = fit_linear_calibration_model(inlier_measurements)

        # compute the residuals and use them to find outliers
        residuals = linear_calibration.get_calibrated_residuals(inlier_measurements)
        res_stddev = np.std(residuals)
        outlier_mask = np.where(
            np.abs(residuals) > inlier_stddev_threshold * res_stddev
        )[0]

        # visualize the inliers and outliers
        if show_outlier_rejection:
            _plot_inliers_and_outliers(inlier_measurements, outlier_mask)

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
        inlier_measurements = [
            x for idx, x in enumerate(inlier_measurements) if idx not in outlier_mask
        ]

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
