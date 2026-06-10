from stereo_matching import StereoOutput
from src.camera import StereoCameraLive
from src.depth.neural import StereoPipelineWrapper
import depthai as dai
import cv2
from typing import cast
import numpy as np


def main():
    depthai_pipeline = dai.Pipeline()
    camera = StereoCameraLive(depthai_pipeline, 640, 400, 30)
    stereo_pipeline = StereoPipelineWrapper(model="raft-stereo")

    camera.stereo.setRectification(True)

    colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

    with depthai_pipeline:
        depthai_pipeline.start()
        max_disparity = 1

        while depthai_pipeline.isRunning():
            left_image, right_image = camera.next_rectified_pair()
            left_image, right_image = left_image.getCvFrame(), right_image.getCvFrame()
            if left_image.ndim == 2 and right_image.ndim == 2:
                left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

            result = stereo_pipeline(left_image, right_image)
            result = cast(StereoOutput, result)

            disparity = result.disparity
            max_disparity = max(max_disparity, np.max(disparity))

            colorizedDisparity = cv2.applyColorMap(
                ((disparity / max_disparity) * 255).astype(np.uint8), colorMap
            )
            cv2.imshow("disparity", colorizedDisparity)
            key = cv2.waitKey(1)
            if key == ord("q"):
                depthai_pipeline.stop()
                break


if __name__ == "__main__":
    main()
