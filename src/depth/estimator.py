from __future__ import annotations
import stereo_matching as sm

import numpy.typing as npt
from abc import ABC, abstractmethod

from typing import Optional, Union, List

from line_profiler import profile as time_profile
from memory_profiler import profile as memory_profile

import numpy as np


class DepthEstimator(ABC):
    @classmethod
    def factory(
        cls, disparity_algorithm: str, device: Optional[str] = None, **kwargs
    ) -> DepthEstimator:
        if disparity_algorithm in NeuralNetworkEstimator.MODELS:
            return NeuralNetworkEstimator(disparity_algorithm, device, **kwargs)
        else:
            raise NotImplementedError("Must implement classical algorithms pipeline")

    @abstractmethod
    def disparity(
        self, left_image: npt.NDArray, right_image: npt.NDArray
    ) -> npt.NDArray:
        pass


class NeuralNetworkEstimator(DepthEstimator):
    MODELS = [
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
