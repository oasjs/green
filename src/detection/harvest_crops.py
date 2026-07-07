# =============================================================================
# harvest_crops.py
# -----------------------------------------------------------------------------
# Turns a folder of ordinary full-scene frames into a sortable pile of cone
# candidate crops, using the SAME colour-segmentation + geometric-filter
# stages already in cone_detection_svm.py. No staged/individual cone photos
# needed -- just point it at normal footage.
#
# Two modes:
#
#   harvest  -- runs the pipeline over every image in --images_dir and saves
#               every surviving candidate bbox as its own crop, under:
#                   out_dir/<color>/candidates/<frame_stem>_<idx>.png
#
#   review   -- opens each harvested crop one at a time and lets you sort it
#               with a single keypress:
#                   p = positive (real cone)     -> out_dir/<color>/pos/
#                   n = negative (false positive) -> out_dir/<color>/neg/
#                   s = skip (leave in candidates/, decide later)
#                   q = quit review early
#
# After review, out_dir/<color>/{pos,neg}/ is exactly the layout
# train_svm_models.py expects.
#
# Usage:
#   python harvest_crops.py harvest --images_dir ./frames --out_dir ./dataset
#   python harvest_crops.py review  --out_dir ./dataset --color yellow
# =============================================================================

from __future__ import annotations

import argparse
import glob
import os

import cv2

from cone_detection_svm import (
    preprocess,
    build_masks,
    clean_mask,
    find_cone_contours,
    filter_contours_geometry,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
COLORS = ["yellow", "blue", "orange"]

# Extra margin added around each tight bounding box before cropping.
# Keeps a bit of context around the silhouette edge, which is what HOG
# will actually see at inference time (the geometric filter's boxes are
# rarely pixel-perfect on the cone's true outline).
BBOX_PADDING_FRAC = 0.12


def _list_images(directory: str) -> list[str]:
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(paths)


def _pad_bbox(bbox: tuple[int, int, int, int],
              frame_w: int, frame_h: int,
              frac: float = BBOX_PADDING_FRAC) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad_w = int(w * frac)
    pad_h = int(h * frac)

    x0 = max(x - pad_w, 0)
    y0 = max(y - pad_h, 0)
    x1 = min(x + w + pad_w, frame_w)
    y1 = min(y + h + pad_h, frame_h)

    return x0, y0, x1 - x0, y1 - y0


# -----------------------------------------------------------------------------
# harvest mode
# -----------------------------------------------------------------------------

def harvest(images_dir: str, out_dir: str, colors: list[str]) -> None:
    image_paths = _list_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for color in colors:
        os.makedirs(os.path.join(out_dir, color, "candidates"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, color, "pos"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, color, "neg"), exist_ok=True)

    total_saved = {color: 0 for color in colors}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] could not read {img_path}")
            continue

        frame_h, frame_w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        hsv = preprocess(img)
        masks = build_masks(hsv)
        frame_counts = {}

        for color in colors:
            mask = masks.get(color)
            if mask is None:
                continue

            clean = clean_mask(mask, color)
            contours = find_cone_contours(clean)
            candidates = filter_contours_geometry(contours, frame_h, color)
            frame_counts[color] = len(candidates)

            for idx, c in enumerate(candidates):
                x, y, w, h = _pad_bbox(c["bbox"], frame_w, frame_h)
                crop = img[y:y + h, x:x + w]
                if crop.size == 0:
                    continue

                out_path = os.path.join(
                    out_dir, color, "candidates", f"{stem}_{idx:03d}.png"
                )
                cv2.imwrite(out_path, crop)
                total_saved[color] += 1

        print(f"[{stem}] " + ", ".join(
            f"{c}: {frame_counts.get(c, 0)}" for c in colors
        ))

    print("\nHarvest complete:")
    for color in colors:
        print(f"  {color}: {total_saved[color]} candidate crops -> "
              f"{os.path.join(out_dir, color, 'candidates')}")
    print("\nNext: run `review` mode to sort candidates into pos/ and neg/.")


# -----------------------------------------------------------------------------
# review mode (interactive sorter)
# -----------------------------------------------------------------------------

def review(out_dir: str, color: str, zoom: int = 4) -> None:
    candidates_dir = os.path.join(out_dir, color, "candidates")
    pos_dir = os.path.join(out_dir, color, "pos")
    neg_dir = os.path.join(out_dir, color, "neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    crop_paths = _list_images(candidates_dir)
    if not crop_paths:
        print(f"No candidates left to review in {candidates_dir}")
        return

    print(f"Reviewing {len(crop_paths)} '{color}' candidates.")
    print("Keys:  [p] positive   [n] negative   [s] skip   [q] quit")

    window = f"review: {color}"
    for i, path in enumerate(crop_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        preview = cv2.resize(img, (w * zoom, h * zoom),
                              interpolation=cv2.INTER_NEAREST)
        cv2.putText(preview, f"{i + 1}/{len(crop_paths)}  p=pos n=neg s=skip q=quit",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow(window, preview)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            os.rename(path, os.path.join(pos_dir, os.path.basename(path)))
        elif key == ord('n'):
            os.rename(path, os.path.join(neg_dir, os.path.basename(path)))
        # 's' or anything else -> leave in candidates/, move to next

    cv2.destroyAllWindows()
    print("Review session ended.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest and sort cone training crops")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_harvest = sub.add_parser("harvest", help="Extract candidate crops from full-scene images")
    p_harvest.add_argument("--images_dir", required=True)
    p_harvest.add_argument("--out_dir", default="./dataset")
    p_harvest.add_argument("--colors", nargs="+", default=COLORS)

    p_review = sub.add_parser("review", help="Interactively sort crops into pos/neg")
    p_review.add_argument("--out_dir", default="./dataset")
    p_review.add_argument("--color", required=True, choices=COLORS)
    p_review.add_argument("--zoom", type=int, default=4)

    args = parser.parse_args()

    if args.mode == "harvest":
        harvest(args.images_dir, args.out_dir, args.colors)
    elif args.mode == "review":
        review(args.out_dir, args.color, args.zoom)


if __name__ == "__main__":
    main()