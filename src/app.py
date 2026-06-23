from src.depth.estimator import DepthEstimator
import cv2
import numpy as np
import depthai as dai
from .camera import StereoCameraLive, StereoCameraRecorded

import numpy.typing as npt


class App:
    class DepthMapVisualizer:
        def __init__(self):
            self._color_map = cv2.applyColorMap(
                np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET
            )
            self._color_map[0] = [0, 0, 0]
            self._max_disparity = 1

        def show(self, disparity: npt.NDArray):
            max_disparity = max(self._max_disparity, np.max(disparity))
            cv2.imshow(
                "disparity",
                cv2.applyColorMap(
                    ((disparity / max_disparity) * 255).astype(np.uint8),
                    self._color_map,
                ),
            )

        @property
        def color_map(self):
            return self._color_map

    def __init__(self, live_cam: bool = True):
        pass

    def run_live(
        self,
        disparity_algorithm: str,
        image_width: int = 1280,
        image_height: int = 800,
        fps: int = 30,
    ):
        pipeline = dai.Pipeline()
        camera = StereoCameraLive(
            pipeline, image_width, image_height, fps, top_crop_ratio=0.4
        )
        depth_estimator = DepthEstimator.factory(disparity_algorithm, camera)
        visualizer = self.DepthMapVisualizer()

        with pipeline:
            pipeline.start()

            while pipeline.isRunning():
                left_image, right_image = camera.next_rectified_pair()
                if left_image.ndim == 2 and right_image.ndim == 2:
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

                visualizer.show(depth_estimator.disparity(left_image, right_image))

                key = cv2.waitKey(1)
                if key == ord("q"):
                    pipeline.stop()
                    break

    def run_recorded(self):
        raise NotImplementedError
