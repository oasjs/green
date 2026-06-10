from typing import Optional, Union, List
from stereo_matching import pipeline, StereoOutput

import numpy as np


class StereoPipelineWrapper:
    def __init__(
        self, model: Optional[str] = None, device: Optional[str] = None, **kwargs
    ):
        self._pipe = pipeline(
            "stereo-matching", model=model, device=device, kwargs=kwargs
        )

    def __call__(
        self,
        left_images: Union[str, "Image.Image", np.ndarray, List],
        right_images: Union[str, "Image.Image", np.ndarray, List],
        batch_size: int = 1,
        colorize: bool = True,
        colormap: str = "turbo",
        focal_length: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> Union[StereoOutput, List[StereoOutput]]:
        return self._pipe(
            left_images,
            right_images,
            batch_size,
            colorize,
            colormap,
            focal_length,
            baseline,
        )
