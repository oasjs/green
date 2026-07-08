"""Live stereo camera source, wrapping the provided StereoCameraLive (DepthAI)."""

import cv2
import numpy as np

# Import the user's existing class — assumed available on PYTHONPATH.
from camera import StereoCameraLive  # type: ignore


class LiveStereoSource:
    """Thin adapter so the app only depends on `get_pair()`, not on DepthAI directly."""

    def __init__(self, camera: StereoCameraLive):
        self._camera = camera

    @classmethod
    def from_config(
        cls,
        pipeline,
        image_width: int,
        image_height: int,
        fps: int,
        top_crop_ratio: float = 0.0,
        image_type=None,
        resize_mode=None,
    ) -> "LiveStereoSource":
        import depthai as dai

        kwargs = dict(
            pipeline=pipeline,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            top_crop_ratio=top_crop_ratio,
        )
        if image_type is not None:
            kwargs["image_type"] = image_type
        else:
            kwargs["image_type"] = dai.ImgFrame.Type.GRAY8
        if resize_mode is not None:
            kwargs["resize_mode"] = resize_mode
        else:
            kwargs["resize_mode"] = dai.ImgResizeMode.CROP

        return cls(StereoCameraLive(**kwargs))

    def get_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (left, right) rectified frames as numpy arrays. True grayscale —
        this is what should be fed to the depth estimators."""
        left, right = self._camera.next_rectified_pair()
        # Ensure 3-channel RGB for consistency with Gradio's Image component.
        if left.ndim == 2:
            left = np.stack([left] * 3, axis=-1)
        if right.ndim == 2:
            right = np.stack([right] * 3, axis=-1)
        return left, right

    def get_center_frame(self) -> np.ndarray:
        """Return the central camera's frame as an RGB numpy array. This one is a
        real color sensor (unlike the mono left/right pair)."""
        frame_bgr = self._camera.next_center_frame()
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    @property
    def camera(self) -> StereoCameraLive:
        return self._camera
