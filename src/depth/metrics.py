from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class AccuracyMetrics:
    mean_absolute_error: np.float64
    root_mean_squarred_error: np.float64 
    bad_pixel_rate_1: np.float64
    bad_pixel_rate_3: np.float64
    valid_ratio: np.float64
    mean_depth_error: np.float64
    min_depth_error: np.float64
    max_depth_error: np.float64

@dataclass
class DisparityMetrics:
    mean_absolute_error: np.float64
    root_mean_squarred_error: np.float64 
    bad_pixel_rate_1: np.float64
    bad_pixel_rate_3: np.float64
    valid_ratio: np.float64

@dataclass
class DepthMetrics:
    mean_absolute_error: np.float64
    min_absolute_error: np.float64
    max_absolute_error: np.float64


class DisparityBenchmark:
    def __init__(self) -> None:
        pass

    # Must print, save and plot the metrics
    def run(self, estimated_disparity_map: npt.NDArray, reference_disparity_map: npt.NDArray) -> AccuracyMetrics:

        mean_abs_error = np.mean(
            np.abs(estimated_disparity_map - reference_disparity_map)
        )
        root_mean_squarred_error = np.sqrt(
            np.mean(estimated_disparity_map - reference_disparity_map)
        )
        bad_pixel_rate_1 = np.mean(
            np.abs(estimated_disparity_map - reference_disparity_map) > 1
        )
        bad_pixel_rate_3 = np.mean(
            np.abs(estimated_disparity_map - reference_disparity_map) > 3
        )

        valid_ratio = np.mean(estimated_disparity_map > 0)

        ground_truth_depth_map = np.array([])
        estimated_depth_map = np.array([])

        depth_error = np.abs(estimated_depth_map - ground_truth_depth_map)
        mean_depth_error = depth_error.mean()
        min_depth_error = np.min(depth_error) 
        max_depth_error = np.max(depth_error)
        # TODO: relative_depth_error =

        return AccuracyMetrics(mean_abs_error, root_mean_squarred_error, bad_pixel_rate_1, bad_pixel_rate_3, valid_ratio, mean_depth_error, min_depth_error, max_depth_error)

