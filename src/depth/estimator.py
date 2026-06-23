from __future__ import annotations
from src.camera import StereoCameraLive
import stereo_matching as sm

import numpy.typing as npt
from abc import ABC, abstractmethod

from typing import Optional, Union, List, cast

from line_profiler import profile as time_profile
from memory_profiler import profile as memory_profile

import numpy as np
import cv2
import depthai as dai


class DepthEstimator(ABC):
    NN_MODELS = [
        "raft-stereo",
        "raft-stereo-middlebury",
        "raft-stereo-eth3d",
        "raft-stereo-realtime",
        "crestereo",
        "aanet",
        "aanet-kitti2012",
        "aanet-sceneflow",
        "foundation-stereo",
        "foundation-stereo-large",
        "unimatch",
        "unimatch-mixdata",
        "unimatch-sceneflow",
        "unimatch-kitti15",
        "unimatch-middlebury",
        "igev-plusplus",
        "igev-plusplus-kitti2012",
        "igev-plusplus-kitti2015",
        "igev-plusplus-middlebury",
        "igev-plusplus-eth3d",
        "igev-stereo",
        "igev-stereo-sceneflow",
        "igev-stereo-kitti2012",
        "igev-stereo-kitti2015",
        "igev-stereo-middlebury",
        "igev-stereo-eth3d",
    ]

    CLASSICAL_ALGORITHMS = ["bm", "sgbm", "sgbm-depthai"]

    @classmethod
    def factory(
        cls,
        disparity_algorithm: str,
        camera: Optional[StereoCameraLive],
        device: Optional[str] = None,
        **kwargs,
    ) -> DepthEstimator:
        if disparity_algorithm in NeuralNetworkEstimator.NN_MODELS:
            return NeuralNetworkEstimator(disparity_algorithm, device, **kwargs)
        else:
            match disparity_algorithm:
                case "bm":
                    return OpenCvBm()
                case "sgbm":
                    return OpenCvSgbm()
                case "sgbm-depthai":
                    if camera is None:
                        raise RuntimeError(
                            "sgbm-depthai requires a StereoCameraLive parameter"
                        )
                    return DepthAiSgbm(camera)
                case _:
                    raise RuntimeError("Must specify classical algorithm")

    @abstractmethod
    @time_profile
    @memory_profile
    def disparity(
        self, left_image: npt.NDArray, right_image: npt.NDArray
    ) -> npt.NDArray:
        pass


class OpenCvBm(DepthEstimator):
    def __init__(self):
        self._stereo = cv2.StereoBM.create()

        block_size = 10
        self.stereo.setBlockSize(block_size * 2 + 1)
        self._num_disparities = 112
        self.stereo.setNumDisparities(self._num_disparities)
        self._min_disparity = 16
        self.stereo.setMinDisparity(self._min_disparity)
        self.stereo.setBlockSize(17)
        self.stereo.setDisp12MaxDiff(0)
        self.stereo.setSpeckleRange(32)
        self.stereo.setSpeckleWindowSize(100)

        smaller_block_size = [n for n in range(5, 256, 2)]
        self._stereo.setSmallerBlockSize(smaller_block_size[0])
        pre_filter_type = [0, 1]
        self._stereo.setPreFilterType(pre_filter_type[0])
        pre_filter_size = [n for n in range(5, 256, 2)]
        self._stereo.setPreFilterSize(pre_filter_size[0])

        texture_threshold = [n for n in range(0, 256)]
        self._stereo.setTextureThreshold(texture_threshold[0])
        uniqueness_ratio = [n for n in range(0, 16)]  # Could be 0 to 100
        self._stereo.setUniquenessRatio(uniqueness_ratio[0])

    def disparity(
        self, left_image: npt.NDArray, right_image: npt.NDArray
    ) -> npt.NDArray:
        left_image, right_image = (
            cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY).astype(np.uint8),
        )
        print(left_image, right_image)
        disparity = self._stereo.compute(left_image, right_image).astype(np.float32)
        return (disparity / 16.0 - self._min_disparity) / self._num_disparities

    @property
    def stereo(self) -> cv2.StereoMatcher:
        return self._stereo


class OpenCvSgbm(OpenCvBm):
    """Uses Birchfield-Tomasi transform by default"""

    def __init__(self):
        self._num_disparities = 1 * 16
        self._min_disparity = 16
        self._stereo = cv2.StereoSGBM.create()
        self._stereo.setNumDisparities(self._num_disparities)
        self._stereo.setMinDisparity(self._min_disparity)
        self._stereo.setBlockSize(6 * 2 + 5)
        # self._stereo.setPreFilterCap()
        self._stereo.setDisp12MaxDiff(0)
        self._stereo.setSpeckleRange(32)
        self._stereo.setSpeckleWindowSize(100)


class DepthAiSgbm(DepthEstimator):
    """Uses Census transform by default"""

    def __init__(self, camera: StereoCameraLive):
        # camera.stereo.initialConfig.setSubpixelFractionalBits(5)
        # camera.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        camera.stereo.initialConfig.setConfidenceThreshold(0)
        # camera.stereo.initialConfig.postProcessing.speckleFilter.enable = True
        # camera.stereo.initialConfig.postProcessing.speckleFilter.speckleRange = 96

        # camera.stereo.initialConfig.postProcessing.temporalFilter.enable = True
        # camera.stereo.initialConfig.postProcessing.temporalFilter.alpha = (
        #     0.4  # Weight of current frame (0-1)
        # )
        # camera.stereo.initialConfig.postProcessing.temporalFilter.delta = (
        #     3  # Threshold for valid depth change
        # )

        # camera.stereo.initialConfig.postProcessing.spatialFilter.enable = True
        # camera.stereo.initialConfig.postProcessing.spatialFilter.alpha = (
        #     0.8  # Edge-preserving strength
        # )
        # camera.stereo.initialConfig.postProcessing.spatialFilter.delta = (
        #     16  # Threshold for valid depth change
        # )
        # camera.stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = (
        #     8  # Radius for hole filling
        # )
        # camera.stereo.initialConfig.postProcessing.spatialFilter.numIterations = (
        #     1  # Number of iterations
        # )

        # camera.stereo.initialConfig.postProcessing.brightnessFilter.minBrightness = (
        #     0  # Minimum brightness threshold
        # )
        # camera.stereo.initialConfig.postProcessing.brightnessFilter.maxBrightness = (
        #     200  # Maximum brightness threshold
        # )

        camera.stereo.setLeftRightCheck(True)
        camera.stereo.setExtendedDisparity(True)
        # camera.stereo.setSubpixel(True)
        self._disparity_queue = camera.stereo.disparity.createOutputQueue()

    def disparity(
        self, left_image: npt.NDArray, right_image: npt.NDArray
    ) -> npt.NDArray:
        return cast(dai.ImgFrame, self._disparity_queue.get()).getFrame()


class NeuralNetworkEstimator(DepthEstimator):
    def __init__(
        self, model: Optional[str] = None, device: Optional[str] = None, **kwargs
    ):
        self._pipeline = sm.pipeline(
            "stereo-matching", model=model, device=device, kwargs=kwargs
        )

    @time_profile
    @memory_profile
    def disparity(
        self,
        left_images: Union[str, "Image.Image", np.ndarray, List],
        right_images: Union[str, "Image.Image", np.ndarray, List],
        batch_size: int = 1,
        colorize: bool = True,
        colormap: str = "turbo",
        focal_length: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> Union[sm.StereoOutput, List[sm.StereoOutput]]:
        output = self._pipeline(
            left_images,
            right_images,
            batch_size,
            colorize,
            colormap,
            focal_length,
            baseline,
        )
        if isinstance(output, sm.StereoOutput):
            return output.disparity
        else:
            return output
