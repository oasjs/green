from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from pprint import pprint

import numpy as np
import yaml
from numpy.typing import NDArray


class StereoCamera:
    class Model(Enum):
        OAKDLR = 1

    @dataclass
    class Info:
        image_width: int
        image_height: int
        intrinsics: NDArray[np.float64]
        distortion_coefficients: NDArray[np.float64]
        rectification_matrix: NDArray[np.float64]
        projection_matrix: NDArray[np.float64]

    def __init__(
        self,
        model: StereoCamera.Model,
        left_camera_info_filepath: Path = Path(__file__).resolve().parent.parent
        / "config/oakdlr_left.yaml",
        right_camera_info_filepath: Path = Path(__file__).resolve().parent.parent
        / "config/oakdlr_right.yaml",
        load_info_from_camera: bool = False,
    ) -> None:
        self._model = model
        self._camera_info: list[StereoCamera.Info | None] = [None, None]

        if load_info_from_camera:
            self._load_info_from_camera()
        else:
            self._load_info_from_yaml(
                left_camera_info_filepath, right_camera_info_filepath
            )

        pprint(self._camera_info)

    def _load_info_from_camera(self) -> None:
        pass

    def _load_info_from_yaml(
        self, left_camera_info_filepath: Path, right_camera_info_filepath: Path
    ) -> None:
        for idx, file_path in enumerate(
            [left_camera_info_filepath, right_camera_info_filepath],
        ):
            with open(file_path) as file:
                yaml_info = yaml.load(file, Loader=yaml.FullLoader)

                image_height = yaml_info["image_height"]
                image_width = yaml_info["image_width"]
                intrinsics = np.array(yaml_info["camera_matrix"]["data"])
                distortion_coefficients = np.array(
                    yaml_info["distortion_coefficients"]["data"]
                )
                rectification_matrix = np.array(
                    yaml_info["rectification_matrix"]["data"]
                )
                projection_matrix = np.array(yaml_info["projection_matrix"]["data"])

                self._camera_info[idx] = StereoCamera.Info(
                    image_width,
                    image_height,
                    intrinsics,
                    distortion_coefficients,
                    rectification_matrix,
                    projection_matrix,
                )

        if (
            not (self.left.image_width and self.right.image_width)
            and self.left.image_width != self.right.image_width
        ) or (
            not (self.left.image_height and self.right.image_height)
            and self.left.image_height != self.right.image_height
        ):
            raise AssertionError(
                "Left and right images must have the same non-zero dimensions. "
                + f"Found left: {self.left.image_width}x{self.left.image_height}, right: {self.right.image_width}x{self.right.image_height}."
            )

    @property
    def model(self) -> StereoCamera.Model:
        return self._model

    @property
    def left(self) -> StereoCamera.Info:
        if self._camera_info[0] is None:
            raise AssertionError("Failed to load information for left camera.")
        return self._camera_info[0]

    @property
    def right(self) -> StereoCamera.Info:
        if self._camera_info[1] is None:
            raise AssertionError("Failed to load information for right camera.")
        return self._camera_info[1]


if __name__ == "__main__":
    oakldr_camera = StereoCamera(StereoCamera.Model.OAKDLR)
