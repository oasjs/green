# =============================================================================
# train_svm_models.py
# -----------------------------------------------------------------------------
# Offline training script: builds one HOG+SVM model per cone colour.
#
# Expected dataset layout (crops, not full frames -- e.g. exported from
# FSOCO bounding-box annotations, or manually cropped):
#
#   dataset/
#     yellow/
#       pos/*.png      <- crops centred on yellow cones
#       neg/*.png      <- hard negatives: background, other cones, track edges
#     blue/
#       pos/*.png
#       neg/*.png
#     orange/
#       pos/*.png
#       neg/*.png
#
# Usage:
#   python train_svm_models.py --dataset ./dataset --out ./models
#
# Output:
#   ./models/yellow_svm.pkl
#   ./models/blue_svm.pkl
#   ./models/orange_svm.pkl
# =============================================================================

from __future__ import annotations

import argparse
import os

from svm_hog_utils import load_dataset, train_svm, save_model, HOG_RESIZE

COLORS = ["yellow", "blue", "orange"]


def train_one_color(dataset_root: str, out_dir: str, color: str) -> None:
    pos_dir = os.path.join(dataset_root, color, "pos")
    neg_dir = os.path.join(dataset_root, color, "neg")

    if not (os.path.isdir(pos_dir) and os.path.isdir(neg_dir)):
        print(f"[skip] {color}: expected folders not found "
              f"({pos_dir}, {neg_dir})")
        return

    print(f"\n=== Training {color} SVM ===")
    X, y = load_dataset(pos_dir, neg_dir, size=HOG_RESIZE)
    print(f"{color}: {int(y.sum())} positive / {int((1 - y).sum())} negative crops")

    model = train_svm(X, y)
    save_model(model, os.path.join(out_dir, f"{color}_svm.pkl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HOG+SVM cone classifiers")
    parser.add_argument("--dataset", default="./dataset",
                         help="Root folder containing per-colour pos/neg crop subfolders")
    parser.add_argument("--out", default="./models",
                         help="Where to save trained .pkl models")
    parser.add_argument("--colors", nargs="+", default=COLORS,
                         help="Subset of colours to train (default: all)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for color in args.colors:
        train_one_color(args.dataset, args.out, color)

    print("\nDone. Point the main pipeline's MODEL_PATHS at the .pkl files above.")


if __name__ == "__main__":
    main()