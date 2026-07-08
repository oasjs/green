"""Cone detection module (Formula SAE: blue / yellow cones).

Mirrors the interface used by `estimators.py` for depth (name, param_spec(),
compute(...)) so it plugs into the same kind of Gradio tab, but detectors
run on a single color image (the center camera) rather than a stereo pair.

Detector.compute(image, params) -> (annotated_bgr_or_rgb_image, detections)
where `detections` is a list of dicts: {"label": str, "bbox": (x, y, w, h), "score": float}
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Shared interface
# =============================================================================

class ConeDetector(Protocol):
    name: str
    available: bool

    def param_spec(self) -> dict[str, tuple[float, float, float, float]]:
        ...

    def compute(
        self, image: np.ndarray, params: dict
    ) -> tuple[np.ndarray, list[dict]]:
        ...


COLOR_RGB = {
    "yellow": (230, 210,  20),
    "blue":   (20,  90,  220),
    "orange": (235, 120,  15),
}


def draw_detections(image: np.ndarray, results: dict[str, list[dict]]) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of an RGB image."""
    out = image.copy()
    for label, candidates in results.items():
        color = COLOR_RGB.get(label, (200, 200, 200))
        for c in candidates:
            x, y, w, h = c["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                out, label, (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out


def summarize(detections: list[dict]) -> str:
    if not detections:
        return "0 cone(s) detected"
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    lines = [f"{len(detections)} cone(s) detected"]
    for label, n in counts.items():
        lines.append(f"  {label}: {n}")
    return "\n".join(lines)


# =============================================================================
# Classical CV pipeline (HSV + contours) — adapted from the user's
# cone_det.py. Display/debug calls (cv2.imshow, matplotlib) removed so it
# runs headless inside the Gradio app; the HSV ranges and morphology are
# unchanged, with the fixed thresholds now exposed as adjustable params.
# =============================================================================

class ClassicalCVConeDetector:
    """HSV + contour-filter pipeline, adapted from the user's `cone_det.py`.

    Display/debug calls (cv2.imshow, matplotlib) from the original were
    removed so it runs headless inside the Gradio app; the HSV ranges and
    morphology are unchanged, with the fixed thresholds now exposed as
    adjustable params.
    """

    name = "Classical CV (HSV + contours)"
    available = True
    requires_images = True

    YELLOW_LOW  = np.array([ 18, 120,  80])
    YELLOW_HIGH = np.array([ 35, 255, 255])

    BLUE_LOW  = np.array([ 95, 120,  50])
    BLUE_HIGH = np.array([135, 255, 255])

    # Base minimum solidity/extent per colour, scaled by the solidity_scale /
    # extent_scale params. Looser than a solid blob since the middle stripe
    # still creates a slight waist even after bridging.
    SOLIDITY_MIN_BASE = {"yellow": 0.55, "blue": 0.50}
    EXTENT_MIN_BASE = {"yellow": 0.35, "blue": 0.35}

    # Stripe colors: blue cones have a white stripe, yellow cones a black stripe.
    STRIPE_WHITE_LOW  = np.array([  0,   0, 180])
    STRIPE_WHITE_HIGH = np.array([179,  50, 255])
    STRIPE_BLACK_LOW  = np.array([  0,   0,   0])
    STRIPE_BLACK_HIGH = np.array([179, 255,  60])

    def param_spec(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            "gamma": (0.3, 3.0, 1.0, 0.1),
            "stripe_bridge": (5, 150, 45, 5),
            "min_area": (10, 3000, 150, 10),
            "max_aspect": (0.3, 2.0, 1.2, 0.05),
            "solidity_scale": (0.5, 1.5, 1.0, 0.05),
            "extent_scale": (0.5, 1.5, 1.0, 0.05),
            "require_stripe": (0, 1, 1, 1),
        }

    # ---- public entry point ----
    def compute(self, image: np.ndarray, params: dict) -> tuple[np.ndarray, list[dict]]:
        # `image` is RGB (Gradio convention); the pipeline was written for BGR.
        img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        results = self._detect_cones(
            img_bgr,
            gamma=params.get("gamma", 1.0),
            stripe_bridge=int(params.get("stripe_bridge", 45)),
            min_area=params.get("min_area", 150),
            max_aspect=params.get("max_aspect", 1.2),
            solidity_scale=params.get("solidity_scale", 1.0),
            extent_scale=params.get("extent_scale", 1.0),
            require_stripe=bool(params.get("require_stripe", 1)),
        )
        annotated = draw_detections(image, results)
        flat = [
            {"label": label, **c}
            for label, cands in results.items()
            for c in cands
        ]
        return annotated, flat

    # ---- pipeline stages ----
    def _detect_cones(
        self,
        img_bgr: np.ndarray,
        gamma: float,
        stripe_bridge: int,
        min_area: float,
        max_aspect: float,
        solidity_scale: float,
        extent_scale: float,
        require_stripe: bool,
    ) -> dict[str, list[dict]]:
        hsv = self._preprocess(img_bgr, gamma)
        masks = self._build_masks(hsv)

        results: dict[str, list[dict]] = {"yellow": [], "blue": []}
        for color, mask in masks.items():
            clean = self._clean_mask(mask, stripe_bridge)
            contours = self._find_cone_contours(clean)
            candidates = self._filter_contours_geometry(
                contours, color, min_area, max_aspect, solidity_scale, extent_scale
            )
            if require_stripe:
                candidates = [c for c in candidates if self._has_stripe(c["bbox"], hsv, color)]
            results[color] = candidates
        return results

    def _percentile_stretch(self, img_bgr: np.ndarray, percentile: float = 95) -> np.ndarray:
        """White-balance via per-channel percentile stretch. Must run before HSV."""
        img = img_bgr.astype(np.float32)
        for i in range(3):
            p = np.percentile(img[:, :, i], percentile)
            if p > 0:
                img[:, :, i] = img[:, :, i] * (255.0 / p)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _apply_gamma_correction(self, src: np.ndarray, gamma: float) -> np.ndarray:
        if gamma <= 0:
            gamma = 1.0
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(src, table)

    def _preprocess(self, img_bgr: np.ndarray, gamma: float) -> np.ndarray:
        """White balance -> gamma -> BGR->HSV. Returns an HSV image."""
        img = self._percentile_stretch(img_bgr)
        img = self._apply_gamma_correction(img, gamma)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def _build_masks(self, hsv: np.ndarray) -> dict[str, np.ndarray]:
        mask_yellow = cv2.inRange(hsv, self.YELLOW_LOW, self.YELLOW_HIGH)
        mask_blue = cv2.inRange(hsv, self.BLUE_LOW, self.BLUE_HIGH)
        return {"yellow": mask_yellow, "blue": mask_blue}

    def _clean_mask(self, mask: np.ndarray, stripe_bridge: int) -> np.ndarray:
        """Opening (kill noise) -> stripe-bridging close -> mild dilation.

        The stripe (white on blue cones, black on yellow) splits each cone's
        color mask into a top blob and a bottom blob. A normal small closing
        kernel doesn't span that gap, so contour extraction returns two
        separate boxes per cone. `stripe_bridge` is a *tall, narrow* closing
        kernel (tall enough to jump the stripe, narrow so it doesn't fuse two
        side-by-side cones together) applied to reconnect top+bottom into one blob.
        """
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Narrow width keeps it from bridging separate cones; height does the work.
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(1, stripe_bridge)))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_bridge)
        mask = cv2.dilate(mask, k_dilate, iterations=1)
        return mask

    def _find_cone_contours(self, mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def _filter_contours_geometry(
        self,
        contours: list,
        color: str,
        min_area: float,
        max_aspect: float,
        solidity_scale: float,
        extent_scale: float,
    ) -> list[dict]:
        """Reject contours that are geometrically incompatible with a cone
        silhouette:
          1. Area bounds     — removes dust and full-frame blobs
          2. Aspect ratio     — cones are taller than wide (w/h < max_aspect)
          3. Solidity         — contour area / convex-hull area; low solidity
                                 means a fragmented/L-shaped/noisy blob
          4. Extent           — contour area / bounding-box area; low extent
                                 means a thin diagonal streak, not a cone

        `solidity_scale`/`extent_scale` multiply the per-colour base thresholds
        (SOLIDITY_MIN_BASE / EXTENT_MIN_BASE) so they can be loosened/tightened
        from the UI without touching per-colour constants.
        """
        sol_min = self.SOLIDITY_MIN_BASE.get(color, 0.65) * solidity_scale
        ext_min = self.EXTENT_MIN_BASE.get(color, 0.40) * extent_scale

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w == 0:
                continue

            aspect = w / h
            if aspect > max_aspect:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < sol_min:
                continue

            extent = area / (w * h)
            if extent < ext_min:
                continue

            candidates.append({"contour": cnt, "hull": hull, "bbox": (x, y, w, h)})
        return candidates

    def _has_stripe(self, bbox: tuple[int, int, int, int], hsv_frame: np.ndarray, color: str) -> bool:
        """Positive evidence check: blue cones have a white stripe, yellow
        cones a black stripe.
        """
        x, y, w, h = bbox
        roi = hsv_frame[y:y + h, x:x + w]
        if roi.size == 0:
            return False

        if color == "blue":
            stripe_mask = cv2.inRange(roi, self.STRIPE_WHITE_LOW, self.STRIPE_WHITE_HIGH)
        else:  # yellow
            stripe_mask = cv2.inRange(roi, self.STRIPE_BLACK_LOW, self.STRIPE_BLACK_HIGH)

        # Stripe should cover a meaningful fraction of the box, not just a few
        # stray pixels — scales with box size instead of a fixed pixel count.
        min_stripe_px = 0.02 * w * h
        return int(np.count_nonzero(stripe_mask)) > min_stripe_px


# =============================================================================
# HOG + SVM — not implemented yet.
# =============================================================================

HOG_SVM_AVAILABLE = True

# Point this at a local FSOCO-style export: a folder containing an `images/`
# subfolder and either YOLO-format labels (`labels/<image_stem>.txt`, one
# `class cx cy w h` line per box, all normalized 0-1) or a single COCO-format
# `annotations.json`. Class names are matched case-insensitively against
# "yellow"/"blue" substrings (e.g. "yellow_cone", "blue_cone" both work).
DEFAULT_CONE_DATASET_DIR: str | None = "FSOCO-MIT.v3-no-augmentation.coco/"


class HogSvmConeDetector:
    """Sliding-window HOG + linear-SVM cone detector, trained on a real
    labelled dataset (e.g. FSOCO) rather than self-generated labels.

    Expects a dataset directory laid out as a Roboflow COCO export:
        <dataset_dir>/train/*.jpg
        <dataset_dir>/train/_annotations.coco.json
        <dataset_dir>/valid/*.jpg + _annotations.coco.json
        <dataset_dir>/test/*.jpg  + _annotations.coco.json
      (only the `splits` passed to __init__ are used for training, "train"
      by default)

      or, as a flat fallback:
        <dataset_dir>/images/*.jpg
        <dataset_dir>/labels/*.txt        (YOLO format, same stem as image)
      or
        <dataset_dir>/images/*.jpg
        <dataset_dir>/annotations.json    (COCO format)

    Positives are box crops pulled straight from the annotations; negatives
    are random crops sampled from the same images that don't overlap any
    annotated box (IoU < 0.1). Both are resized to `WIN_SIZE`, turned into
    HOG features, and used to fit a `sklearn.svm.LinearSVC`. The trained
    model is cached to `<dataset_dir>/hog_svm_model.joblib` so training only
    happens once per dataset, not once per process.

    Training happens lazily on the first `compute()` call (or eagerly if you
    call `train()` yourself), using `dataset_dir` passed to `__init__` or
    `DEFAULT_CONE_DATASET_DIR` if set. Color (blue vs yellow) is assigned
    post-detection from the same HSV masks the classical detector uses — the
    SVM itself is cone-vs-not-cone.
    """

    name = "HOG + SVM"
    available = HOG_SVM_AVAILABLE
    requires_images = True

    # Fixed patch size fed to the HOG descriptor. Cones are taller than
    # wide, so the winSize follows suit; must be a multiple of the block
    # stride below (8x8 cells, 16x16 blocks).
    WIN_SIZE = (32, 64)  # (w, h)
    BLOCK_SIZE = (16, 16)
    BLOCK_STRIDE = (8, 8)
    CELL_SIZE = (8, 8)
    NBINS = 9

    MODEL_FILENAME = "hog_svm_model.joblib"

    def __init__(
        self,
        dataset_dir: str | None = None,
        max_train_images: int | None = 400,
        splits: tuple[str, ...] = ("train",),
    ) -> None:
        self.dataset_dir = dataset_dir or DEFAULT_CONE_DATASET_DIR
        self.max_train_images = max_train_images
        # Which Roboflow split folder(s) to train on. Defaults to just
        # "train"; pass e.g. ("train", "valid") to pull in more data.
        self.splits = splits
        self._svm = None
        self._hog = cv2.HOGDescriptor(
            self.WIN_SIZE, self.BLOCK_SIZE, self.BLOCK_STRIDE, self.CELL_SIZE, self.NBINS
        )

    def param_spec(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            "conf_threshold": (-2.0, 2.0, 0.2, 0.05),
            "step_size": (4, 32, 8, 2),
            "scale_step": (1.05, 1.5, 1.15, 0.05),
            "min_scale": (0.5, 1.0, 0.6, 0.05),
            "max_scale": (1.0, 3.0, 1.8, 0.1),
            "nms_iou": (0.1, 0.9, 0.3, 0.05),
        }

    # ---- public entry point ----
    def compute(self, image: np.ndarray, params: dict) -> tuple[np.ndarray, list[dict]]:
        img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        if self._svm is None:
            self.train()

        conf_threshold = params.get("conf_threshold", 0.2)
        step_size = int(params.get("step_size", 8))
        scale_step = params.get("scale_step", 1.15)
        min_scale = params.get("min_scale", 0.6)
        max_scale = params.get("max_scale", 1.8)
        nms_iou = params.get("nms_iou", 0.3)

        boxes, scores = self._sliding_window_detect(
            img_bgr, conf_threshold, step_size, scale_step, min_scale, max_scale
        )
        boxes, scores = self._nms(boxes, scores, nms_iou)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        results: dict[str, list[dict]] = {"yellow": [], "blue": []}
        for (x, y, w, h), score in zip(boxes, scores):
            label = self._classify_color(hsv, (x, y, w, h))
            if label is None:
                continue
            results[label].append({"bbox": (x, y, w, h), "score": float(score)})

        annotated = draw_detections(image, results)
        flat = [{"label": label, **c} for label, cands in results.items() for c in cands]
        return annotated, flat

    # ---- dataset loading ----
    @staticmethod
    def _class_to_label(class_name: str) -> str | None:
        name = class_name.lower()
        if "yellow" in name:
            return "yellow"
        if "blue" in name:
            return "blue"
        return None

    def _load_yolo_annotations(self, images_dir, labels_dir) -> list[tuple]:
        """Returns [(image_path, [(label, x, y, w, h_px), ...]), ...] in pixel coords."""
        import os

        samples = []
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue
            img_path = os.path.join(images_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            boxes = []
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id, cx, cy, w, h = parts[:5]
                    label = self._class_to_label(cls_id)
                    if label is None:
                        # class file may use numeric ids only; caller can
                        # override _class_to_label / pass a classes.txt map.
                        continue
                    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                    bw, bh = w * w_img, h * h_img
                    bx, by = cx * w_img - bw / 2, cy * h_img - bh / 2
                    boxes.append((label, int(bx), int(by), int(bw), int(bh)))
            if boxes:
                samples.append((img_path, boxes))
        return samples

    def _load_coco_annotations(self, images_dir, ann_path) -> list[tuple]:
        import json
        import os

        with open(ann_path) as f:
            coco = json.load(f)

        cat_to_label = {
            c["id"]: self._class_to_label(c["name"]) for c in coco.get("categories", [])
        }
        img_id_to_info = {im["id"]: im for im in coco.get("images", [])}

        boxes_by_img: dict[int, list[tuple]] = {}
        for ann in coco.get("annotations", []):
            label = cat_to_label.get(ann["category_id"])
            if label is None:
                continue
            x, y, w, h = ann["bbox"]
            boxes_by_img.setdefault(ann["image_id"], []).append((label, int(x), int(y), int(w), int(h)))

        samples = []
        for img_id, boxes in boxes_by_img.items():
            info = img_id_to_info.get(img_id)
            if info is None:
                continue
            img_path = os.path.join(images_dir, info["file_name"])
            if os.path.exists(img_path):
                samples.append((img_path, boxes))
        return samples

    def _load_dataset(self) -> list[tuple]:
        import os

        if not self.dataset_dir:
            raise NotImplementedError(
                "HOG + SVM detector needs a labelled cone dataset. Pass "
                "dataset_dir=... to HogSvmConeDetector(...) or set "
                "detectors.DEFAULT_CONE_DATASET_DIR to a folder containing "
                "images/ plus YOLO labels/ or an annotations.json (COCO), or "
                "a Roboflow-style export with train/valid/test split folders."
            )

        # --- Roboflow COCO export layout: <dataset_dir>/<split>/*.jpg plus
        # <dataset_dir>/<split>/_annotations.coco.json, images and json living
        # in the same folder (no separate images/ subfolder). ---
        split_samples = []
        for split in self.splits:
            split_dir = os.path.join(self.dataset_dir, split)
            ann_path = os.path.join(split_dir, "_annotations.coco.json")
            if os.path.isdir(split_dir) and os.path.exists(ann_path):
                split_samples.extend(self._load_coco_annotations(split_dir, ann_path))

        if split_samples:
            samples = split_samples
        else:
            # --- Fallback: flat images/ + labels/ (YOLO) or images/ +
            # annotations.json (COCO). ---
            images_dir = os.path.join(self.dataset_dir, "images")
            labels_dir = os.path.join(self.dataset_dir, "labels")
            coco_path = os.path.join(self.dataset_dir, "annotations.json")

            if not os.path.isdir(images_dir):
                raise NotImplementedError(
                    f"Couldn't find a train/valid/test split (with "
                    f"_annotations.coco.json) or an images/ folder under "
                    f"{self.dataset_dir!r}."
                )

            if os.path.isdir(labels_dir):
                samples = self._load_yolo_annotations(images_dir, labels_dir)
            elif os.path.exists(coco_path):
                samples = self._load_coco_annotations(images_dir, coco_path)
            else:
                raise NotImplementedError(
                    f"No labels/ (YOLO) or annotations.json (COCO) found under {self.dataset_dir!r}."
                )

        if not samples:
            raise NotImplementedError(
                f"Found images under {self.dataset_dir!r} but no usable blue/yellow "
                "cone annotations. Check that class names contain 'yellow'/'blue', "
                "or adjust `_class_to_label`."
            )

        if self.max_train_images:
            samples = samples[: self.max_train_images]
        return samples

    # ---- training ----
    def train(self, force: bool = False) -> None:
        """Load the dataset, extract HOG features, fit the SVM. Cached to
        disk under the dataset directory unless `force=True`."""
        import os

        cache_name = self.MODEL_FILENAME
        if self.dataset_dir and self.splits != ("train",):
            stem, ext = os.path.splitext(self.MODEL_FILENAME)
            cache_name = f"{stem}.{'-'.join(self.splits)}{ext}"
        cache_path = os.path.join(self.dataset_dir, cache_name) if self.dataset_dir else None
        if not force and cache_path and os.path.exists(cache_path):
            import joblib

            self._svm = joblib.load(cache_path)
            return

        samples = self._load_dataset()

        rng = np.random.default_rng(0)
        pos_feats, neg_feats = [], []

        for img_path, boxes in samples:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            pixel_boxes = [(x, y, w, h) for _, x, y, w, h in boxes]

            for _, x, y, w, h in boxes:
                if w <= 0 or h <= 0:
                    continue
                pos_feats.append(self._hog_features(img, (x, y, w, h)))

            # A few random non-overlapping negatives per image.
            neg_target = 3
            attempts = 0
            got = 0
            asp = self.WIN_SIZE[1] / self.WIN_SIZE[0]
            while got < neg_target and attempts < 50:
                attempts += 1
                w = int(rng.integers(20, max(21, w_img // 6)))
                h = int(w * asp)
                if w >= w_img or h >= h_img:
                    continue
                x = int(rng.integers(0, w_img - w))
                y = int(rng.integers(0, h_img - h))
                if any(self._iou((x, y, w, h), pb) > 0.1 for pb in pixel_boxes):
                    continue
                neg_feats.append(self._hog_features(img, (x, y, w, h)))
                got += 1

        if len(pos_feats) < 3 or len(neg_feats) < 3:
            raise NotImplementedError(
                f"Not enough usable training samples from {self.dataset_dir!r} "
                f"({len(pos_feats)} positives, {len(neg_feats)} negatives)."
            )

        X = np.array(pos_feats + neg_feats)
        y = np.array([1] * len(pos_feats) + [0] * len(neg_feats))

        from sklearn.svm import LinearSVC

        svm = LinearSVC(C=1.0, max_iter=5000)
        svm.fit(X, y)
        self._svm = svm

        if cache_path:
            import joblib

            joblib.dump(svm, cache_path)

    # ---- inference ----
    def _hog_features(self, img_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        patch = img_bgr[max(0, y):y + h, max(0, x):x + w]
        if patch.size == 0:
            patch = np.zeros((self.WIN_SIZE[1], self.WIN_SIZE[0], 3), dtype=np.uint8)
        patch = cv2.resize(patch, self.WIN_SIZE, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return self._hog.compute(gray).ravel()

    def _sliding_window_detect(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float,
        step_size: int,
        scale_step: float,
        min_scale: float,
        max_scale: float,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        win_w, win_h = self.WIN_SIZE
        h_img, w_img = img_bgr.shape[:2]

        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []

        scale = min_scale
        while scale <= max_scale:
            w = int(win_w * scale)
            h = int(win_h * scale)
            if w < 8 or h < 8 or w >= w_img or h >= h_img:
                scale *= scale_step
                continue

            for y in range(0, h_img - h, step_size):
                for x in range(0, w_img - w, step_size):
                    feat = self._hog_features(img_bgr, (x, y, w, h))
                    score = float(self._svm.decision_function(feat.reshape(1, -1))[0])
                    if score >= conf_threshold:
                        boxes.append((x, y, w, h))
                        scores.append(score)
            scale *= scale_step

        return boxes, scores

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _nms(
        self, boxes: list[tuple[int, int, int, int]], scores: list[float], iou_thresh: float
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        if not boxes:
            return [], []
        order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        keep: list[int] = []
        while order:
            i = order.pop(0)
            keep.append(i)
            order = [j for j in order if self._iou(boxes[i], boxes[j]) < iou_thresh]
        return [boxes[i] for i in keep], [scores[i] for i in keep]

    def _classify_color(
        self, hsv: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> str | None:
        x, y, w, h = bbox
        roi = hsv[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        yellow_px = int(np.count_nonzero(
            cv2.inRange(roi, ClassicalCVConeDetector.YELLOW_LOW, ClassicalCVConeDetector.YELLOW_HIGH)
        ))
        blue_px = int(np.count_nonzero(
            cv2.inRange(roi, ClassicalCVConeDetector.BLUE_LOW, ClassicalCVConeDetector.BLUE_HIGH)
        ))
        min_px = 0.05 * w * h
        if yellow_px < min_px and blue_px < min_px:
            return None
        return "yellow" if yellow_px >= blue_px else "blue"


# =============================================================================
# YOLOv26 — cone-trained weights, auto-downloaded from Hugging Face.
# =============================================================================

import json
import os
import urllib.request

try:
    from ultralytics import YOLO  # type: ignore

    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import onnxruntime  # type: ignore # noqa: F401

    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ONNXRUNTIME_AVAILABLE = False


# Repo hosting the cone-trained weight files. Both .pt (PyTorch) and .onnx
# (ONNX Runtime) files are supported — pass a filename ending in either
# extension as `weights_name`, or a matching URL as `weights_url`.
HF_REPO_ID = "FiremelonX8/yolo_training"
HF_REPO_API_URL = f"https://huggingface.co/api/models/{HF_REPO_ID}"
HF_RESOLVE_URL_TMPL = "https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"

DEFAULT_WEIGHTS_FILENAME = "yolo26_m_09-2.pt"
SUPPORTED_WEIGHTS_EXTENSIONS = (".pt", ".onnx")

WEIGHTS_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "cone_detector_weights"
)


def list_available_weights(repo_id: str = HF_REPO_ID) -> list[str]:
    """List `.pt`/`.onnx` weight files available in the Hugging Face repo.

    Requires network access. Falls back to just the known default filename
    if the API call fails for any reason (offline, rate-limited, etc.).
    """
    try:
        with urllib.request.urlopen(HF_REPO_API_URL, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        siblings = data.get("siblings", [])
        names = [
            s["rfilename"] for s in siblings
            if s.get("rfilename", "").endswith(SUPPORTED_WEIGHTS_EXTENSIONS)
        ]
        return names or [DEFAULT_WEIGHTS_FILENAME]
    except Exception:
        return [DEFAULT_WEIGHTS_FILENAME]


def _download_weights(
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    repo_id: str = HF_REPO_ID,
    cache_dir: str = WEIGHTS_CACHE_DIR,
) -> str:
    """Download a weights file (by name) from the Hugging Face repo, caching
    it locally so subsequent calls don't re-fetch it.

    Returns the local filesystem path to the weights file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dest_path = os.path.join(cache_dir, filename)
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        url = HF_RESOLVE_URL_TMPL.format(repo_id=repo_id, filename=filename)
        tmp_path = dest_path + ".part"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
    return dest_path


def _download_from_url(url: str, cache_dir: str = WEIGHTS_CACHE_DIR) -> str:
    """Download a weights file from an arbitrary URL, caching it locally by
    filename. Works with any direct-download link (Hugging Face `resolve`
    links, S3, etc.) — not limited to `HF_REPO_ID`.

    Returns the local filesystem path to the weights file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Strip query string (e.g. "?download=true") to get a clean filename.
    parsed_name = url.split("/")[-1].split("?")[0] or "weights.pt"
    dest_path = os.path.join(cache_dir, parsed_name)
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        tmp_path = dest_path + ".part"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
    return dest_path


class YoloConeDetector:
    name = "YOLOv26"
    available = YOLO_AVAILABLE
    requires_images = True

    MODEL_VARIANTS = ["yolov26n.pt", "yolov26s.pt"]

    def __init__(
        self,
        weights_path: str | None = None,
        weights_name: str | None = None,
        repo_id: str = HF_REPO_ID,
        weights_url: str | None = None,
    ):
        """
        Args:
            weights_path: Explicit local path to a .pt or .onnx file. Highest
                priority.
            weights_name: Filename of a weights file (.pt or .onnx) in the
                Hugging Face repo. Downloaded and cached lazily on first
                `compute()` call. Use `list_available_weights()` to see
                what's available.
            repo_id: Hugging Face repo to pull `weights_name` from.
            weights_url: Any direct-download URL to a .pt or .onnx file
                (e.g. a Hugging Face "resolve/main/..." link). Downloaded
                and cached by filename. Takes priority over `weights_name`
                but not over `weights_path`. Loading a .onnx file requires
                `onnxruntime` to be installed.
            If none of `weights_path`, `weights_name`, or `weights_url` are
            given, `DEFAULT_WEIGHTS_FILENAME` from `repo_id` is used.
        """
        self._weights_path = weights_path
        self._weights_name = weights_name
        self._repo_id = repo_id
        self._weights_url = weights_url
        self._model = None

    def set_weights(self, weights_name: str, repo_id: str | None = None) -> None:
        """Switch to a different weights file from the Hugging Face repo.

        Takes effect on the next `compute()` call (the model is reloaded
        lazily). Clears any previously set explicit `weights_path`/`weights_url`.
        """
        self._weights_path = None
        self._weights_url = None
        self._weights_name = weights_name
        if repo_id is not None:
            self._repo_id = repo_id
        self._model = None

    def set_weights_url(self, url: str) -> None:
        """Switch to a weights file downloaded directly from `url`.

        Accepts any direct-download link, e.g. a Hugging Face
        "https://huggingface.co/<repo>/resolve/main/<file>.pt" URL. Takes
        effect on the next `compute()` call. Clears any previously set
        `weights_path`/`weights_name`.
        """
        self._weights_path = None
        self._weights_name = None
        self._weights_url = url.strip()
        self._model = None

    def set_weights_path(self, path: str) -> None:
        """Switch to a local weights file at `path` (e.g. from a file picker
        upload). Takes effect on the next `compute()` call. Clears any
        previously set `weights_name`/`weights_url`.
        """
        self._weights_path = path.strip()
        self._weights_name = None
        self._weights_url = None
        self._model = None

    def current_weights_label(self) -> str:
        """Human-readable description of what weights will be (or are) loaded."""
        if self._weights_path:
            return f"local file: {self._weights_path}"
        if self._weights_url:
            return f"URL: {self._weights_url}"
        return f"HF repo {self._repo_id}: {self._weights_name or DEFAULT_WEIGHTS_FILENAME}"

    def param_spec(self) -> dict[str, tuple[float, float, float, float]]:
        return {
            "conf_threshold": (0.05, 0.95, 0.35, 0.05),
            "iou_threshold": (0.1, 0.9, 0.45, 0.05),
        }

    # Classes the model can output that we deliberately don't track (not a
    # naming mismatch — the model is correctly saying "not sure" or "not a
    # standard boundary/marker cone"). Logged at DEBUG, not flagged as a
    # warning like a genuinely unrecognized class name would be.
    IGNORED_CLASS_SUBSTRINGS = ("unknown",)

    @staticmethod
    def _normalize_label(raw_cls_name: str) -> str | None:
        """Map a model's raw class name to 'yellow', 'blue', or 'orange', if
        possible.

        Handles variants like 'yellow_cone', 'cone-yellow', 'Yellow', etc. by
        substring match rather than requiring an exact name. 'orange' covers
        both small and large orange cones (e.g. 'orange_cone' and
        'large_orange_cone') — both map to the same 'orange' bucket since
        the bbox size already distinguishes them if you need that.
        """
        name = raw_cls_name.lower()
        if "yellow" in name:
            return "yellow"
        if "blue" in name:
            return "blue"
        if "orange" in name:
            return "orange"
        return None

    def compute(self, image: np.ndarray, params: dict) -> tuple[np.ndarray, list[dict]]:
        if not YOLO_AVAILABLE:
            raise NotImplementedError(
                "YOLOv26 cone detector needs `ultralytics` installed (pip "
                "install ultralytics)."
            )
        if self._model is None:
            if self._weights_path:
                weights_path = self._weights_path
            elif self._weights_url:
                weights_path = _download_from_url(self._weights_url)
            else:
                # Lazily download (and cache) the requested weights file
                # from the Hugging Face repo on first use.
                weights_path = _download_weights(
                    filename=self._weights_name or DEFAULT_WEIGHTS_FILENAME,
                    repo_id=self._repo_id,
                )

            is_onnx = weights_path.lower().endswith(".onnx")
            if is_onnx and not ONNXRUNTIME_AVAILABLE:
                raise NotImplementedError(
                    f"Weights file {weights_path!r} is an ONNX model, which "
                    "needs `onnxruntime` installed (pip install onnxruntime, "
                    "or onnxruntime-gpu for CUDA)."
                )

            backend = "ONNX Runtime" if is_onnx else "PyTorch"
            logger.info(
                "YoloConeDetector: loading weights from %s (backend: %s)",
                weights_path, backend,
            )
            self._model = YOLO(weights_path)
            logger.info(
                "YoloConeDetector: model loaded. class names = %s", self._model.names
            )

        conf_threshold = params.get("conf_threshold", 0.35)
        iou_threshold = params.get("iou_threshold", 0.45)
        logger.debug(
            "YoloConeDetector: running predict on image shape=%s dtype=%s "
            "conf=%.3f iou=%.3f",
            getattr(image, "shape", None), getattr(image, "dtype", None),
            conf_threshold, iou_threshold,
        )

        result = self._model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]

        raw_boxes = list(result.boxes)
        logger.info(
            "YoloConeDetector: %d raw box(es) above conf=%.3f before label filtering",
            len(raw_boxes), conf_threshold,
        )

        results: dict[str, list[dict]] = {"yellow": [], "blue": [], "orange": []}
        ignored_labels: dict[str, int] = {}
        unrecognized_labels: dict[str, int] = {}
        for box in raw_boxes:
            raw_cls_name = result.names[int(box.cls[0])]
            label = self._normalize_label(raw_cls_name)
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            # Log every raw box's class, score, and bbox at INFO — this is
            # what you need to cross-reference "the box at this location"
            # against what's actually in the image, to tell a mislabeled
            # class apart from an actually-orange cone.
            logger.info(
                "YoloConeDetector: raw box class=%r score=%.3f bbox=%s -> "
                "mapped label=%s",
                raw_cls_name, score, bbox, label,
            )
            if label is None:
                name_lower = raw_cls_name.lower()
                if any(s in name_lower for s in self.IGNORED_CLASS_SUBSTRINGS):
                    ignored_labels[raw_cls_name] = ignored_labels.get(raw_cls_name, 0) + 1
                else:
                    unrecognized_labels[raw_cls_name] = unrecognized_labels.get(raw_cls_name, 0) + 1
                continue
            results[label].append({"bbox": bbox, "score": score})

        if ignored_labels:
            logger.debug(
                "YoloConeDetector: %d box(es) intentionally not tracked "
                "(unknown_cone or similar): %s",
                sum(ignored_labels.values()), ignored_labels,
            )
        if unrecognized_labels:
            logger.warning(
                "YoloConeDetector: %d box(es) skipped because their class "
                "name couldn't be mapped to 'yellow'/'blue'/'orange': %s. "
                "_normalize_label() matches by substring — update it if "
                "your classes use different naming, e.g. numeric ids.",
                sum(unrecognized_labels.values()), unrecognized_labels,
            )
        if not raw_boxes:
            logger.warning(
                "YoloConeDetector: model returned zero boxes at conf=%.3f. "
                "Try lowering conf_threshold, and confirm the image being "
                "passed in isn't blank/black and matches the training "
                "resolution/preprocessing the model expects.",
                conf_threshold,
            )

        annotated = draw_detections(image, results)
        flat = [{"label": label, **c} for label, cands in results.items() for c in cands]
        logger.info(
            "YoloConeDetector: final detections — yellow=%d blue=%d orange=%d",
            len(results["yellow"]), len(results["blue"]), len(results["orange"]),
        )
        return annotated, flat
