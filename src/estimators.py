"""Depth/disparity estimator classes: BM, SGBM, RAFT-Stereo."""

from abc import ABC, abstractmethod
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
 
try:
    from stereo_matching import pipeline as stereo_matching_pipeline  # type: ignore
    RAFT_AVAILABLE = True
except Exception:
    RAFT_AVAILABLE = False
 
try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    MONO_AVAILABLE = True
except Exception:
    MONO_AVAILABLE = False
 
try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except Exception:
    DEPTHAI_AVAILABLE = False


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def colorize(disp: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(disp, 2), np.percentile(disp, 98)
    d = np.clip((disp - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return (cm.get_cmap("magma")(d)[:, :, :3] * 255).astype(np.uint8)


class DepthEstimator(ABC):
    """Common interface for all disparity estimators."""

    name: str = "base"
    requires_images: bool = True  # False for on-device estimators that ignore left/right
    requires_right: bool = True  # False for monocular estimators that only need `left`

    @abstractmethod
    def compute(self, left: np.ndarray, right: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (colorized_disparity, raw_disparity)."""

    @staticmethod
    @abstractmethod
    def param_spec() -> dict:
        """Describe tunable params: {key: (min, max, default, step)}."""


class BMEstimator(DepthEstimator):
    name = "StereoBM"

    @staticmethod
    def param_spec():
        return {
            "num_disparities": (16, 256, 64, 16),
            "block_size": (5, 51, 15, 2),
            "pre_filter_cap": (1, 63, 31, 1),
            "uniqueness_ratio": (0, 30, 15, 1),
            "speckle_window": (0, 200, 100, 1),
            "speckle_range": (0, 32, 32, 1),
            "texture_threshold": (0, 30, 10, 1),
        }

    def compute(self, left, right, params):
        l, r = to_gray(left), to_gray(right)
        num_disp = max(16, (int(params["num_disparities"]) // 16) * 16)
        block = int(params["block_size"]) | 1

        stereo = cv2.StereoBM.create(numDisparities=num_disp, blockSize=block)
        stereo.setPreFilterCap(int(params["pre_filter_cap"]))
        stereo.setUniquenessRatio(int(params["uniqueness_ratio"]))
        stereo.setSpeckleWindowSize(int(params["speckle_window"]))
        stereo.setSpeckleRange(int(params["speckle_range"]))
        stereo.setTextureThreshold(int(params["texture_threshold"]))

        disp = stereo.compute(l, r).astype(np.float32) / 16.0
        return colorize(disp), disp


class SGBMEstimator(DepthEstimator):
    name = "StereoSGBM"
    _MODES = {
        "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
        "HH": cv2.STEREO_SGBM_MODE_HH,
        "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }

    @staticmethod
    def param_spec():
        return {
            "min_disparity": (-64, 64, 0, 1),
            "num_disparities": (16, 256, 128, 16),
            "block_size": (1, 21, 5, 2),
            "p1_mult": (1, 32, 8, 1),
            "p2_mult": (1, 128, 32, 1),
            "disp12_max_diff": (-1, 50, 1, 1),
            "pre_filter_cap": (1, 63, 63, 1),
            "uniqueness_ratio": (0, 30, 10, 1),
            "speckle_window": (0, 200, 100, 1),
            "speckle_range": (0, 32, 2, 1),
        }

    def compute(self, left, right, params):
        l, r = to_gray(left), to_gray(right)
        num_disp = max(16, (int(params["num_disparities"]) // 16) * 16)
        block = int(params["block_size"]) | 1
        mode = self._MODES[params.get("mode", "SGBM_3WAY")]

        stereo = cv2.StereoSGBM.create(
            minDisparity=int(params["min_disparity"]),
            numDisparities=num_disp,
            blockSize=block,
            P1=int(params["p1_mult"]) * block ** 2,
            P2=int(params["p2_mult"]) * block ** 2,
            disp12MaxDiff=int(params["disp12_max_diff"]),
            preFilterCap=int(params["pre_filter_cap"]),
            uniquenessRatio=int(params["uniqueness_ratio"]),
            speckleWindowSize=int(params["speckle_window"]),
            speckleRange=int(params["speckle_range"]),
            mode=mode,
        )
        disp = stereo.compute(l, r).astype(np.float32) / 16.0
        return colorize(disp), disp


class RaftStereoEstimator(DepthEstimator):
    """RAFT-Stereo via the `stereo_matching` library (github.com/shriarul5273/stereo_matching).

    Model loading, preprocessing, and colorized-disparity postprocessing are all
    handled by the library's `pipeline()` factory; we just cache one pipeline per
    selected model variant and pass through the recurrent-iteration count.
    """

    name = "RAFT-Stereo"

    # RAFT-Stereo variants registered in the library.
    MODEL_VARIANTS = [
        "raft-stereo",
        "raft-stereo-middlebury",
        "raft-stereo-eth3d",
        "raft-stereo-realtime",
    ]

    def __init__(self):
        self._pipe_cache: dict[str, "object"] = {}

    @staticmethod
    def param_spec():
        return {
            "num_iters": (1, 64, 32, 1),
        }

    def _get_pipe(self, model_id: str):
        if model_id not in self._pipe_cache:
            self._pipe_cache[model_id] = stereo_matching_pipeline(
                "stereo-matching", model=model_id
            )
        return self._pipe_cache[model_id]

    def compute(self, left, right, params):
        if not RAFT_AVAILABLE:
            raise RuntimeError(
                "RAFT-Stereo backend not installed (`pip install stereo_matching`)."
            )
        model_id = params.get("model_id", self.MODEL_VARIANTS[0])
        pipe = self._get_pipe(model_id)
        pipe.model.config.num_iters = int(params.get("num_iters", 32))

        left_img = Image.fromarray(left)
        right_img = Image.fromarray(right)
        result = pipe(left_img, right_img, colorize=True, colormap="magma")
        return result.colored_disparity, np.asarray(result.disparity)


class DepthAiSgbmEstimator(DepthEstimator):
    """On-device stereo matching (DepthAI StereoDepth node). Census transform by default.

    Unlike the other estimators, matching happens on the camera itself, so `compute()`
    ignores the left/right arrays it's given and just reads the latest disparity frame
    off the device queue. Post-processing filter params are pushed live via the node's
    `inputConfig` queue, so sliders in the UI take effect without restarting the pipeline.

    Only usable with a live camera; there is no "upload two images" mode for this one.
    """

    name = "DepthAI-SGBM"
    requires_images = False

    def __init__(self, camera):
        if not DEPTHAI_AVAILABLE:
            raise RuntimeError("depthai package not installed.")

        stereo = camera.stereo

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

        pp = stereo.initialConfig.postProcessing
        pp.speckleFilter.enable = True
        pp.speckleFilter.speckleRange = 50  # was 96: too permissive, ate real detail

        pp.temporalFilter.enable = True
        pp.temporalFilter.alpha = 0.4  # weight of current frame (0-1)
        pp.temporalFilter.delta = 3  # threshold for valid depth change

        pp.spatialFilter.enable = True
        pp.spatialFilter.alpha = 0.5  # was 0.8: too aggressive, over-smoothed edges
        pp.spatialFilter.delta = 20
        pp.spatialFilter.holeFillingRadius = 4  # was 8: less aggressive hole-filling
        pp.spatialFilter.numIterations = 1

        pp.brightnessFilter.minBrightness = 0
        pp.brightnessFilter.maxBrightness = 255  # was 200: was silently dropping bright pixels

        stereo.setLeftRightCheck(True)
        # Extended disparity (close range) and subpixel (precision) fight each other
        # when combined; pick one. Subpixel gives smoother results for most scenes,
        # switch to setExtendedDisparity(True) instead if your subject is very close
        # to the camera and depth is short-ranged.
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        stereo.setSubpixelFractionalBits(3)

        self._stereo = stereo
        self._disparity_queue = stereo.disparity.createOutputQueue()
        # Runtime config updates: push a new StereoDepthConfig while the pipeline is
        # running (no restart needed). Requires DepthAI v3's node input queues.
        self._config_queue = stereo.inputConfig.createInputQueue()
        self._last_params: dict | None = None

    @staticmethod
    def param_spec():
        return {
            "speckle_range": (0, 200, 50, 1),
            "temporal_alpha": (0.0, 1.0, 0.4, 0.05),
            "temporal_delta": (0, 50, 3, 1),
            "spatial_alpha": (0.0, 1.0, 0.5, 0.05),
            "spatial_delta": (0, 50, 20, 1),
            "spatial_hole_filling_radius": (0, 16, 4, 1),
            "spatial_iterations": (1, 4, 1, 1),
            "brightness_min": (0, 255, 0, 1),
            "brightness_max": (0, 255, 255, 1),
        }

    def compute(self, left, right, params):
        # left/right are ignored: the device already matched its own synced frame pair.
        if params and params != self._last_params:
            cfg = self._stereo.initialConfig
            pp = cfg.postProcessing
            pp.speckleFilter.speckleRange = int(params["speckle_range"])
            pp.temporalFilter.alpha = float(params["temporal_alpha"])
            pp.temporalFilter.delta = int(params["temporal_delta"])
            pp.spatialFilter.alpha = float(params["spatial_alpha"])
            pp.spatialFilter.delta = int(params["spatial_delta"])
            pp.spatialFilter.holeFillingRadius = int(params["spatial_hole_filling_radius"])
            pp.spatialFilter.numIterations = int(params["spatial_iterations"])
            pp.brightnessFilter.minBrightness = int(params["brightness_min"])
            pp.brightnessFilter.maxBrightness = int(params["brightness_max"])
            self._config_queue.send(cfg)
            self._last_params = dict(params)

        disp = self._disparity_queue.get().getFrame()
        disp = disp.astype(np.float32)
        return colorize(disp), disp

class MonoDepthEstimator(DepthEstimator):
    """Monocular relative depth via Depth Anything V2 (Hugging Face `transformers` pipeline).
 
    Unlike the stereo estimators, this only needs one image — `right` is accepted
    for interface compatibility but ignored. Output is relative (not metric) depth
    unless you pick one of the metric-finetuned checkpoints.
    """
 
    name = "Depth Anything V2 (mono)"
    requires_right = False
 
    MODEL_VARIANTS = [
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything/Depth-Anything-V2-Base-hf",
        "depth-anything/Depth-Anything-V2-Large-hf",
        # Metric-finetuned: output real depth in meters directly, but only within
        # the domain they were finetuned on.
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    ]
 
    def __init__(self):
        self._pipe_cache: dict[str, "object"] = {}
 
    @staticmethod
    def param_spec():
        return {}
 
    def _get_pipe(self, model_id: str):
        if model_id not in self._pipe_cache:
            self._pipe_cache[model_id] = hf_pipeline(task="depth-estimation", model=model_id)
        return self._pipe_cache[model_id]
 
    def compute(self, left, right, params):
        if not MONO_AVAILABLE:
            raise RuntimeError(
                "Depth Anything V2 backend not installed (`pip install transformers torch`)."
            )
        model_id = params.get("model_id", self.MODEL_VARIANTS[0])
        pipe = self._get_pipe(model_id)
 
        result = pipe(Image.fromarray(left))
        depth = result["predicted_depth"]
        if hasattr(depth, "detach"):  # torch.Tensor
            depth = depth.detach().cpu().numpy()
        depth = np.asarray(depth).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        if depth.shape != left.shape[:2]:
            depth = cv2.resize(depth, (left.shape[1], left.shape[0]))
 
        return colorize(depth), depth
