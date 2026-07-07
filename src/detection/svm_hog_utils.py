# =============================================================================
# svm_hog_utils.py
# -----------------------------------------------------------------------------
# HOG feature extraction + linear-SVM training / persistence / inference
# utilities for the cone-detection pipeline.
#
# This module is the "shape verification" stage that replaces the disabled
# validate_cone_shape() / has_stripe() heuristics in the main pipeline.
#
# Pattern: hand-crafted feature extractor (HOG) + traditional ML classifier
# (linear SVM) -- the classical CV/ML combo, no CNN involved.
#
# Dependencies:
#   pip install scikit-learn scikit-image joblib
# =============================================================================

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# (width, height) every ROI is resized to before HOG extraction.
# Cones are taller than wide -> portrait aspect ratio.
HOG_RESIZE = (64, 128)

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
)

# Decision-function margin required to accept a detection.
# 0.0 = the raw SVM boundary; raising this trades recall for precision.
DEFAULT_SCORE_THRESHOLD = 0.0


@dataclass
class SvmVerdict:
    accepted: bool
    score: float


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def extract_hog_features(roi_bgr_or_gray: cv2.typing.MatLike,
                          size: tuple[int, int] = HOG_RESIZE) -> np.ndarray:
    """
    Convert a cropped cone candidate into a fixed-length HOG feature vector.

    Accepts either a grayscale or BGR ROI; BGR is converted internally.
    Resizing to a fixed size is mandatory -- HOG's output length depends on
    input dimensions, and the SVM expects a constant-length vector.
    """
    if roi_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(roi_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr_or_gray

    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    features = hog(resized, **HOG_PARAMS)
    return features


# -----------------------------------------------------------------------------
# Dataset loading (offline training)
# -----------------------------------------------------------------------------

def _load_crops_from_dir(directory: str, size: tuple[int, int]) -> np.ndarray:
    """Load every image in `directory`, extract HOG features, stack into a matrix."""
    paths = sorted(
        glob.glob(os.path.join(directory, "*.png")) +
        glob.glob(os.path.join(directory, "*.jpg")) +
        glob.glob(os.path.join(directory, "*.jpeg"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")

    feats = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        feats.append(extract_hog_features(img, size))

    return np.vstack(feats)


def load_dataset(pos_dir: str, neg_dir: str,
                  size: tuple[int, int] = HOG_RESIZE
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for one colour class from two folders of image crops:
      pos_dir -> positive examples of that cone colour (label 1)
      neg_dir -> hard negatives / background crops (label 0)

    Expected layout, e.g. for yellow:
        dataset/yellow/pos/*.png
        dataset/yellow/neg/*.png
    """
    pos_feats = _load_crops_from_dir(pos_dir, size)
    neg_feats = _load_crops_from_dir(neg_dir, size)

    X = np.vstack([pos_feats, neg_feats])
    y = np.concatenate([
        np.ones(len(pos_feats), dtype=int),
        np.zeros(len(neg_feats), dtype=int),
    ])
    return X, y


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_svm(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
              random_state: int = 42, C: float = 1.0):
    """
    Train a StandardScaler + LinearSVC pipeline and print a held-out
    classification report so you can sanity-check precision/recall before
    trusting the model on-track.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=C, class_weight="balanced", max_iter=10000),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["background", "cone"]))

    return model


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model -> {path}")


def load_model(path: str):
    return joblib.load(path)


# -----------------------------------------------------------------------------
# Inference (used inside the detection pipeline)
# -----------------------------------------------------------------------------

def verify_with_svm(bbox: tuple[int, int, int, int],
                     frame_bgr: cv2.typing.MatLike,
                     model,
                     size: tuple[int, int] = HOG_RESIZE,
                     score_threshold: float = DEFAULT_SCORE_THRESHOLD
                     ) -> SvmVerdict:
    """
    Crop `bbox` out of the full frame, run HOG + SVM, and return a verdict.

    Uses `decision_function` rather than `predict` so the caller can apply
    a custom margin (e.g. tighten precision by raising score_threshold) and
    optionally use the score to rank/deduplicate overlapping candidates.
    """
    x, y, w, h = bbox
    h_frame, w_frame = frame_bgr.shape[:2]

    # Clamp to frame bounds -- bounding rects near edges can overshoot.
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, w_frame), min(y + h, h_frame)
    if x1 <= x0 or y1 <= y0:
        return SvmVerdict(accepted=False, score=-np.inf)

    roi = frame_bgr[y0:y1, x0:x1]
    feats = extract_hog_features(roi, size).reshape(1, -1)

    score = float(model.decision_function(feats)[0])
    return SvmVerdict(accepted=score >= score_threshold, score=score)