from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast
from pprint import pprint

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
import yaml


class StereoCameraLive:
    def __init__(
        self,
        pipeline: dai.Pipeline,
        image_width: int,
        image_height: int,
        fps: int,
        top_crop_ratio: float = 0.0,
        image_type: dai.ImgFrame.Type = dai.ImgFrame.Type.GRAY8,
        resize_mode: dai.ImgResizeMode = dai.ImgResizeMode.CROP,
    ):
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be greater than 0")
        if fps <= 0:
            raise ValueError("Camera FPS must be greater than 0")

        self._image_width, self._image_height = image_width, image_height
        self._fps = fps

        self._device = (
            pipeline.getDefaultDevice()
        )  # TODO: Change this to get a specific device?

        self._stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self._stereo.setRectification(True)
        self._stereo.setExtendedDisparity(True)  # Shorter min depth, double disp range
        self._stereo.setSubpixel(True)

        left_camera: dai.node.Camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B
        )
        right_camera: dai.node.Camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C
        )

        left_camera.requestOutput(
            size=(image_width, image_height),
            type=image_type,
            resizeMode=resize_mode,
            fps=fps,
        ).link(self._stereo.left)

        right_camera.requestOutput(
            size=(image_width, image_height),
            type=image_type,
            resizeMode=resize_mode,
            fps=fps,
        ).link(self._stereo.right)

        if top_crop_ratio > 0.0:
            print("HEERE")
            self._left_rectified_output = self._attach_crop_manip(
                pipeline,
                self._stereo.rectifiedLeft,
                image_width,
                image_height,
                top_crop_ratio,
            )
            self._right_rectified_output = self._attach_crop_manip(
                pipeline,
                self._stereo.rectifiedRight,
                image_width,
                image_height,
                top_crop_ratio,
            )
        else:
            self._left_rectified_output = self._stereo.rectifiedLeft.createOutputQueue()
            self._right_rectified_output = (
                self._stereo.rectifiedRight.createOutputQueue()
            )

    def _attach_crop_manip(
        self,
        pipeline: dai.Pipeline,
        source: dai.Node.Output,
        width: int,
        height: int,
        top_crop_ratio: float,
    ) -> dai.MessageQueue:
        crop_y = int(height * top_crop_ratio)
        cropped_height = height - int(height * top_crop_ratio)

        manip = pipeline.create(dai.node.ImageManip)
        manip.setBackend(
            dai.node.ImageManip.Backend.GPU
            if self._device.hasGPU()
            else dai.node.ImageManip.Backend.CPU
        )
        manip.initialConfig.addCrop(x=0, y=crop_y, w=width, h=cropped_height)
        source.link(manip.inputImage)
        return manip.out.createOutputQueue()

    @property
    def device(self) -> dai.Device:
        return self._device

    @property
    def stereo(self) -> dai.node.StereoDepth:
        return self._stereo

    def next_rectified_pair(self) -> tuple[npt.NDArray, npt.NDArray]:
        left_rectified_image = cast(dai.ImgFrame, self._left_rectified_output.get())
        right_rectified_image = cast(dai.ImgFrame, self._right_rectified_output.get())
        return left_rectified_image.getCvFrame(), right_rectified_image.getCvFrame()

    def save_info_to_yaml(
        self, left_camera_info_filepath: Path, right_camera_info_filepath: Path
    ) -> None:
        raise NotImplementedError()


class StereoCameraRecorded:
    @dataclass
    class Info:
        intrinsics: npt.NDArray
        distortion: npt.NDArray
        rectification: npt.NDArray
        projection: npt.NDArray
        rectify_map_x: npt.NDArray
        rectify_map_y: npt.NDArray

    def __init__(
        self,
        image_width: int,
        image_height: int,
        left_camera_info_filepath: Path = Path(__file__).resolve().parent.parent
        / "config/oakdlr_left.yaml",
        right_camera_info_filepath: Path = Path(__file__).resolve().parent.parent
        / "config/oakdlr_right.yaml",
    ):
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be greater than 0")

        self._image_width = image_width
        self._image_height = image_height
        self._left_camera_info = self._load_info(left_camera_info_filepath)
        self._right_camera_info = self._load_info(right_camera_info_filepath)

    def next_rectified_pair(
        self, left_image_filepath: Path, right_image_filepath: Path
    ) -> tuple[npt.NDArray, npt.NDArray]:

        rectified_left_image = cv2.remap(
            self._load_image(left_image_filepath),
            self._left_camera_info.rectify_map_x,
            self._left_camera_info.rectify_map_y,
            cv2.INTER_LINEAR,
        )
        rectified_right_image = cv2.remap(
            self._load_image(right_image_filepath),
            self._right_camera_info.rectify_map_x,
            self._right_camera_info.rectify_map_y,
            cv2.INTER_LINEAR,
        )
        return rectified_left_image, rectified_right_image

    def _load_info(self, path: Path) -> StereoCameraRecorded.Info:
        with open(path) as yaml_file:
            yaml_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

            intrinsics = np.array(yaml_info["camera_matrix"]["data"])
            distortion = np.array(yaml_info["distortion_coefficients"]["data"])
            rectification = np.array(yaml_info["rectification_matrix"]["data"])
            projection = np.array(yaml_info["projection_matrix"]["data"])

            rectify_map_x, rectify_map_y = cv2.initUndistortRectifyMap(
                intrinsics,
                distortion,
                rectification,
                projection,
                (self._image_width, self._image_height),
                cv2.CV_32FC1,
            )

            return StereoCameraRecorded.Info(
                intrinsics,
                distortion,
                rectification,
                projection,
                rectify_map_x,
                rectify_map_y,
            )

    def _load_image(self, path: Path) -> npt.NDArray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img


# TODO: Remove this from here. Create a proper test
if __name__ == "__main__":
    pipeline = dai.Pipeline()

    image_width, image_heigth = 640, 400
    fps = 30
    camera = StereoCameraLive(pipeline, image_width, image_heigth, fps)
    camera.stereo.setRectification(True)
    camera.stereo.setExtendedDisparity(True)
    camera.stereo.setLeftRightCheck(True)

    disparityQueue = camera.stereo.disparity.createOutputQueue()

    colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

    with pipeline:
        pipeline.start()
        maxDisparity = 1
        while pipeline.isRunning():
            disparity = cast(dai.ImgFrame, disparityQueue.get())
            npDisparity = disparity.getFrame()
            maxDisparity = max(maxDisparity, np.max(npDisparity))
            colorizedDisparity = cv2.applyColorMap(
                ((npDisparity / maxDisparity) * 255).astype(np.uint8), colorMap
            )
            cv2.imshow("disparity", colorizedDisparity)
            key = cv2.waitKey(1)
            if key == ord("q"):
                pipeline.stop()
                break
