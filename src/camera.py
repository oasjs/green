from __future__ import annotations

from pathlib import Path
from typing import cast
import depthai as dai
import numpy.typing as npt


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
        self._stereo.setLeftRightCheck(True)

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

        # Central color camera — not part of the stereo pair, just a plain RGB feed.
        center_camera: dai.node.Camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )
        self._center_output = center_camera.requestOutput(
            size=(image_width, image_height),
            type=dai.ImgFrame.Type.BGR888i,
            resizeMode=resize_mode,
            fps=fps,
        ).createOutputQueue()

        if top_crop_ratio > 0.0:
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

    def next_center_frame(self) -> npt.NDArray:
        center_image = cast(dai.ImgFrame, self._center_output.get())
        return center_image.getCvFrame()

    def save_calibration_data(self) -> None:
        with self.device as device:
            calibration = device.readCalibration()
            calibration.eepromToJsonFile(
                Path(__file__).parent.parent
                / "data"
                / f"{device.getDeviceInfo().name}-calibration.json"
            )
