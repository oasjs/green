"""No-ground-truth quality metrics for comparing stereo disparity estimators.

Without ground-truth depth, we fall back to self-consistency and proxy-quality
signals computed directly from the stereo pair and the estimated disparity:

- Reprojection error: warp the right image into the left view using the
  estimated disparity, then compare against the actual left image. Large
  error means the disparity doesn't explain the observed parallax.
- Valid-pixel ratio: fraction of pixels with a usable (non-zero, finite)
  disparity estimate. Cheap density proxy.
- Edge alignment: how well disparity discontinuities line up with intensity
  edges in the reference image. Good depth maps break where objects do.
"""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


@dataclass
class QualityMetrics:
    reprojection_l1: float  # mean abs intensity error after warping, lower is better
    reprojection_ssim: float  # structural similarity after warping, higher is better
    valid_ratio: float  # fraction of pixels with usable disparity, higher is better
    edge_alignment: float  # IoU of disparity edges vs image edges, higher is better

    def as_text(self) -> str:
        return (
            f"Reprojection L1: {self.reprojection_l1:.2f}   "
            f"SSIM: {self.reprojection_ssim:.3f}\n"
            f"Valid pixels: {self.valid_ratio * 100:.1f}%   "
            f"Edge alignment: {self.edge_alignment:.3f}"
        )


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _manual_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight single-scale SSIM fallback if scikit-image isn't installed."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = (11, 11)
    mu_a = cv2.GaussianBlur(a, kernel, 1.5)
    mu_b = cv2.GaussianBlur(b, kernel, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, kernel, 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, kernel, 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, kernel, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return float(np.mean(ssim_map))


def _warp_right_to_left(right_gray: np.ndarray, disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample the right image at (x - disparity, y) to synthesize the left view.

    Returns (warped_image, valid_mask) — valid_mask excludes pixels that would
    sample outside the right image's bounds.
    """
    h, w = disparity.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_x = xs - disparity.astype(np.float32)
    valid = (src_x >= 0) & (src_x < w) & (disparity > 0) & np.isfinite(disparity)
    warped = cv2.remap(right_gray, src_x, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped, valid


def _edge_alignment(disparity: np.ndarray, reference_gray: np.ndarray, valid: np.ndarray) -> float:
    disp_filled = np.where(valid, disparity, 0).astype(np.float32)
    disp_norm = cv2.normalize(disp_filled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    disp_edges = cv2.Canny(disp_norm, 50, 150) > 0
    img_edges = cv2.Canny(reference_gray, 50, 150) > 0

    # Dilate image edges a little: disparity edges rarely land on the exact
    # same pixel as intensity edges, so allow a small tolerance band.
    img_edges_dilated = cv2.dilate(img_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    intersection = np.logical_and(disp_edges, img_edges_dilated).sum()
    union = np.logical_or(disp_edges, img_edges_dilated).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute(left: np.ndarray, right: np.ndarray, disparity: np.ndarray) -> QualityMetrics:
    """Compute all no-ground-truth quality metrics for one disparity estimate.

    `disparity` must be the raw (uncolorized) disparity map, same H×W as left/right.
    """
    left_gray = _to_gray(left)
    right_gray = _to_gray(right)

    if disparity.shape != left_gray.shape:
        disparity = cv2.resize(
            disparity.astype(np.float32), (left_gray.shape[1], left_gray.shape[0])
        )

    warped, valid = _warp_right_to_left(right_gray, disparity)
    valid_ratio = float(np.mean(disparity > 0)) if disparity.size else 0.0

    if valid.sum() > 0:
        l1 = float(np.mean(np.abs(left_gray[valid].astype(np.float32) - warped[valid].astype(np.float32))))
        if SKIMAGE_AVAILABLE:
            ssim = float(_ssim(left_gray, warped, data_range=255))
        else:
            ssim = _manual_ssim(left_gray, warped)
    else:
        l1, ssim = float("nan"), float("nan")

    edge_align = _edge_alignment(disparity, left_gray, valid)

    return QualityMetrics(
        reprojection_l1=l1,
        reprojection_ssim=ssim,
        valid_ratio=valid_ratio,
        edge_alignment=edge_align,
    )
