"""
Convert FSOCO COCO-format annotations into cropped pos/ images for SVM training.

Supports two modes:

1. Single split (original behavior):
    python fsoco_to_crops.py --ann annotations.json --images_dir ./fsoco_images --out_dir ./dataset

2. Multiple splits (train/valid/test), auto-discovered under a root folder:
    python fsoco_to_crops.py --split_root ./fsoco_dataset --out_dir ./dataset

   Expects a layout like:
     fsoco_dataset/
       train/
         _annotations.coco.json
         img1.jpg ...
       valid/
         _annotations.coco.json
         img1.jpg ...
       test/
         _annotations.coco.json
         img1.jpg ...

   Each split's annotations file name can be customized with --ann_filename
   (default: "_annotations.coco.json", the Roboflow/FSOCO export default).
   All splits are merged into the same --out_dir/<color>/pos folder, with the
   split name prefixed to filenames to avoid collisions.

Expects FSOCO's COCO export:
    {
      "images": [{"id": 1, "file_name": "img1.jpg", ...}, ...],
      "annotations": [{"image_id": 1, "category_id": 3, "bbox": [x, y, w, h]}, ...],
      "categories": [{"id": 3, "name": "yellow_cone"}, ...]
    }
bbox format is COCO standard: [x_min, y_min, width, height], top-left origin.
"""
import argparse
import json
import os
import cv2

# Map FSOCO category names -> our color folders. Adjust if your export uses different names.
CATEGORY_MAP = {
    "blue_cone": "blue",
    "yellow_cone": "yellow",
    "large_orange_cone": "orange",
    "small_orange_cone": "orange",
}
PAD_FRAC = 0.12  # match harvest_crops.py padding so train/test crops look alike


def pad_bbox(x, y, w, h, img_w, img_h, pad_frac=PAD_FRAC):
    pad_w, pad_h = w * pad_frac, h * pad_frac
    x0 = max(0, int(x - pad_w))
    y0 = max(0, int(y - pad_h))
    x1 = min(img_w, int(x + w + pad_w))
    y1 = min(img_h, int(y + h + pad_h))
    return x0, y0, x1, y1


def process_split(ann_path, images_dir, out_dir, split_name, counts, totals):
    """Process one split's annotations file, accumulating crops into out_dir.

    counts: dict color -> running crop index (shared across splits so filenames
            don't collide and we get a true running total).
    totals: dict color -> running total crop count (for the summary printout).
    """
    if not os.path.isfile(ann_path):
        print(f"  [!] Skipping split '{split_name}': annotations file not found at {ann_path}")
        return 0, 0

    with open(ann_path, "r") as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    categories_by_id = {c["id"]: c["name"] for c in coco["categories"]}

    skipped_unmapped = 0
    skipped_missing_img = 0
    written = 0

    for ann in coco["annotations"]:
        cat_name = categories_by_id.get(ann["category_id"])
        color = CATEGORY_MAP.get(cat_name)
        if color is None:
            skipped_unmapped += 1
            continue

        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue

        img_path = os.path.join(images_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            skipped_missing_img += 1
            continue

        h_img, w_img = img.shape[:2]
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = pad_bbox(x, y, w, h, w_img, h_img)
        if x1 <= x0 or y1 <= y0:
            continue

        crop = img[y0:y1, x0:x1]
        color_out_dir = os.path.join(out_dir, color, "pos")
        os.makedirs(color_out_dir, exist_ok=True)

        idx = counts.get(color, 0)
        base_name = img_info["file_name"].rsplit(".", 1)[0]
        prefix = f"{split_name}_" if split_name else ""
        out_path = os.path.join(color_out_dir, f"{prefix}{base_name}_{idx}.png")
        cv2.imwrite(out_path, crop)

        counts[color] = idx + 1
        totals[color] = totals.get(color, 0) + 1
        written += 1

    print(f"  Split '{split_name}': {written} crops written"
          f"{' (from ' + str(len(coco['annotations'])) + ' annotations)' if coco.get('annotations') else ''}")
    if skipped_unmapped:
        print(f"    Skipped {skipped_unmapped} annotations with unmapped category (check CATEGORY_MAP)")
    if skipped_missing_img:
        print(f"    Skipped {skipped_missing_img} annotations with unreadable/missing image files")

    return skipped_unmapped, skipped_missing_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=None,
                    help="Path to a single FSOCO COCO-format annotations JSON (single-split mode)")
    ap.add_argument("--images_dir", default=None,
                    help="Folder containing the source images (single-split mode)")
    ap.add_argument("--split_root", default=None,
                    help="Root folder containing train/valid/test subfolders (multi-split mode)")
    ap.add_argument("--splits", default="train,valid,test",
                    help="Comma-separated subfolder names to look for under --split_root "
                         "(default: train,valid,test). Missing ones are skipped with a warning.")
    ap.add_argument("--ann_filename", default="_annotations.coco.json",
                    help="Annotations file name expected inside each split folder "
                         "(default: _annotations.coco.json, the Roboflow/FSOCO export default)")
    ap.add_argument("--out_dir", required=True, help="Output dataset root")
    args = ap.parse_args()

    if not args.split_root and not (args.ann and args.images_dir):
        ap.error("Either provide --split_root (multi-split mode), "
                 "or both --ann and --images_dir (single-split mode).")

    counts = {}
    totals = {}

    if args.split_root:
        split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
        print(f"Processing splits {split_names} under {args.split_root} ...")
        for split_name in split_names:
            split_dir = os.path.join(args.split_root, split_name)
            ann_path = os.path.join(split_dir, args.ann_filename)
            # images usually live in the same folder as the annotations file
            process_split(ann_path, split_dir, args.out_dir, split_name, counts, totals)
    else:
        process_split(args.ann, args.images_dir, args.out_dir, split_name="", counts=counts, totals=totals)

    print("Done.")
    for color, n in totals.items():
        print(f"  {color}: {n} crops -> {os.path.join(args.out_dir, color, 'pos')}")


if __name__ == "__main__":
    main()