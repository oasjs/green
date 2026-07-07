# =============================================================================
# Cone Detection Pipeline — Formula Student Classic CV
# + HOG/SVM shape-verification stage
# =============================================================================
#
# This is your original pipeline with the disabled shape/stripe checks
# replaced by a HOG + linear-SVM classifier (see svm_hog_utils.py and
# train_svm_models.py). Everything else (colour segmentation, morphology,
# contour extraction, geometric pre-filter) is unchanged.
#
# Requires trained models produced by train_svm_models.py:
#   ./models/yellow_svm.pkl
#   ./models/blue_svm.pkl
#   ./models/orange_svm.pkl
#
# =============================================================================

from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from svm_hog_utils import verify_with_svm, load_model, HOG_RESIZE

# HSV colour ranges  (OpenCV scale: H 0-179, S 0-255, V 0-255)
YELLOW_LOW  = np.array([ 18, 120,  80])
YELLOW_HIGH = np.array([ 35, 255, 255])

BLUE_LOW    = np.array([ 95, 120,  50])
BLUE_HIGH   = np.array([135, 255, 255])

# Orange wraps around H=0 in OpenCV → two sub-ranges merged later
ORANGE_LOW_A  = np.array([  0, 150,  80])
ORANGE_HIGH_A = np.array([  8, 255, 255])
ORANGE_LOW_B  = np.array([170, 150,  80])
ORANGE_HIGH_B = np.array([179, 255, 255])

# Minimum solidity per colour (blue/yellow have stripes that fragment the blob)
SOLIDITY_MIN = {
    "yellow": 0.55,
    "blue":   0.50,
    "orange": 0.75,
}

# SVM decision-function margin per colour. Raise a colour's threshold if it's
# producing false positives; lower it if it's rejecting real cones.
SVM_SCORE_THRESHOLD = {
    "yellow": 0.0,
    "blue":   0.0,
    "orange": 0.0,
}

MODEL_PATHS = {
    "yellow": "./models/yellow_svm.pkl",
    "blue":   "./models/blue_svm.pkl",
    "orange": "./models/orange_svm.pkl",
}

PATH_TO_IMAGES = '../../assets/images/'


# -----------------------------------------------------------------------------
# Model loading (once, at import/startup time — not per-frame)
# -----------------------------------------------------------------------------

def load_svm_models(paths: dict[str, str] = MODEL_PATHS) -> dict[str, object]:
    """
    Load every available per-colour SVM model. Colours whose .pkl is missing
    are silently skipped (their candidates pass through unverified) so you
    can bring models online one colour at a time during development.
    """
    models = {}
    for color, path in paths.items():
        try:
            models[color] = load_model(path)
            print(f"[svm] loaded {color} model from {path}")
        except FileNotFoundError:
            print(f"[svm] no model found for {color} at {path} — "
                  f"candidates of this colour will NOT be shape-verified")
    return models


# Utilities
def display_image(title: str, image: cv2.typing.MatLike) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(title: str, image: cv2.typing.MatLike) -> None:
    cv2.imwrite(str(title + ".jpg"), image)


def plot_hsv(img_rgb: cv2.typing.MatLike, hsv: cv2.typing.MatLike) -> None:
    pixel_colors = img_rgb.reshape((-1, 3))
    norm = mcolors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(h.flatten(), s.flatten(), z=v.flatten(),
               facecolors=pixel_colors, marker=".")
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    plt.show()


# Pre-processing
def percentile_stretch(img_bgr: cv2.typing.MatLike,
                        percentile: float = 95) -> cv2.typing.MatLike:
    img = img_bgr.astype(np.float32)
    for i in range(3):
        p = np.percentile(img[:, :, i], percentile)
        if p > 0:
            img[:, :, i] = img[:, :, i] * (255.0 / p)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_gamma_correction(src, gamma) -> cv2.typing.MatLike:
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)


def preprocess(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    gamma = 1
    img = percentile_stretch(img)
    img = apply_gamma_correction(img, gamma)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv


# Colour segmentation
def build_masks(hsv: cv2.typing.MatLike) -> dict[str, cv2.typing.MatLike]:
    mask_yellow = cv2.inRange(hsv, YELLOW_LOW,  YELLOW_HIGH)
    mask_blue   = cv2.inRange(hsv, BLUE_LOW,    BLUE_HIGH)

    mask_or_a   = cv2.inRange(hsv, ORANGE_LOW_A, ORANGE_HIGH_A)
    mask_or_b   = cv2.inRange(hsv, ORANGE_LOW_B, ORANGE_HIGH_B)
    mask_orange = cv2.bitwise_or(mask_or_a, mask_or_b)

    return {
        "yellow": mask_yellow,
        "blue":   mask_blue,
        "orange": mask_orange,
    }


# Morphological cleaning
def clean_mask(mask: cv2.typing.MatLike, color: str = "orange") -> cv2.typing.MatLike:
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    k7  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    mask = cv2.dilate(mask, k5, iterations=1)

    return mask


# Contour extraction
def find_cone_contours(mask: cv2.typing.MatLike) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


# Contour filtering
def filter_contours_geometry(contours: list, img_height: int, color: str,
                              min_area: float = 80.0,
                              max_aspect_ratio: float = 1.2) -> list[dict]:
    """
    Cheap geometric pre-filter, run BEFORE the SVM (which is more expensive).

    FIX vs. original: bounding box now comes from the raw contour `cnt`,
    not from `approxPolyN`, which can fail/degenerate on noisy contours.
    FIX: dropped the `if i == 0: continue` — that discarded a valid contour
    for no reason (index 0 is not special for RETR_EXTERNAL output).
    FIX: solidity is now actually computed and used to reject candidates,
    instead of being computed nowhere.
    """
    sol_min = SOLIDITY_MIN.get(color, 0.65)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect_ratio = w / h
        if aspect_ratio > max_aspect_ratio:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < sol_min:
            continue

        extent = area / (w * h)
        if extent < 0.25:
            continue

        candidates.append({
            "contour": cnt,
            "hull": hull,
            "bbox": (x, y, w, h),
            "solidity": solidity,
        })

    return candidates


# Size discrimination (orange only)
def classify_orange_size(candidate: dict, small_max_height: int = 60) -> str:
    _, _, _, h = candidate["bbox"]
    return "orange_small" if h <= small_max_height else "orange_big"


# Visualisation helper
def draw_detections(img: cv2.typing.MatLike,
                     results: dict[str, list[dict]]) -> cv2.typing.MatLike:
    COLOR_BGR = {
        "yellow":       (  0, 220, 220),
        "blue":         (220,  80,   0),
        "orange":       (  0, 140, 255),
        "orange_small": (  0, 140, 255),
        "orange_big":   (  0,  60, 200),
    }

    out = img.copy()
    for label, candidates in results.items():
        bgr = COLOR_BGR.get(label, (200, 200, 200))
        for c in candidates:
            x, y, w, h = c["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
            score_txt = f" ({c['svm_score']:.2f})" if "svm_score" in c else ""
            cv2.putText(out, label + score_txt, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1,
                        cv2.LINE_AA)

    return out


# Main pipeline
def detect_cones(img: cv2.typing.MatLike,
                  hsv: cv2.typing.MatLike,
                  svm_models: dict[str, object]) -> dict[str, list[dict]]:
    """
    Full detection pipeline for a single frame.

    Stage order (cheapest -> most expensive):
      1. Colour segmentation      (region proposal)
      2. Morphological cleaning
      3. Contour extraction
      4. Geometric pre-filter     (area / aspect ratio / solidity / extent)
      5. HOG + SVM verification   (shape confirmation, replaces the old
                                    validate_cone_shape / has_stripe checks)
      6. Orange size split (small/big) for the remaining orange candidates
    """
    masks = build_masks(hsv)

    results: dict[str, list[dict]] = {
        "yellow":       [],
        "blue":         [],
        "orange_small": [],
        "orange_big":   [],
    }

    for color, mask in masks.items():
        clean = clean_mask(mask, color)
        contours = find_cone_contours(clean)
        candidates = filter_contours_geometry(contours, hsv.shape[0], color)

        model = svm_models.get(color)
        verified = []
        for c in candidates:
            if model is not None:
                verdict = verify_with_svm(
                    c["bbox"], img, model,
                    size=HOG_RESIZE,
                    score_threshold=SVM_SCORE_THRESHOLD.get(color, 0.0),
                )
                if not verdict.accepted:
                    continue
                c["svm_score"] = verdict.score
            verified.append(c)

        if color == "orange":
            for c in verified:
                results[classify_orange_size(c)].append(c)
        else:
            results[color] = verified

    return results


def main() -> None:
    svm_models = load_svm_models()

    image_path = 'oak_left_image_raw.png'
    img = cv2.imread(PATH_TO_IMAGES + image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {PATH_TO_IMAGES}{image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = preprocess(img)

    results = detect_cones(img, hsv, svm_models)

    for label, candidates in results.items():
        print(f"[{label}] {len(candidates)} cone(s) detected")

    annotated = draw_detections(img, results)
    save_image("Detections", annotated)
    display_image("Detections", annotated)

    print("Done.")


if __name__ == "__main__":
    main()