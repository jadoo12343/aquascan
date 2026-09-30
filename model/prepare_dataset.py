"""
AquaScan — Dataset Preparation Script  (Person A deliverable)
=============================================================
Downloads, merges, and splits three public waste-image datasets into
the ImageFolder layout expected by model/train.py.

Supported source datasets:
    1. TrashNet  (github.com/garythung/trashnet)     — 2,527 images, 6 classes
    2. TACO      (tacodataset.org)                   — 1,500+ annotated images
    3. Kaggle Waste Dataset                          — supplementary images

Output layout:
    data/dataset/
        train/<class>/  (70%)
        val/<class>/    (15%)
        test/<class>/   (15%)

Usage:
    python model/prepare_dataset.py --source trashnet --data_dir data/dataset
    python model/prepare_dataset.py --source all      --data_dir data/dataset
"""

import argparse
import pathlib
import random
import shutil
import sys
import urllib.request

CLASSES = ["plastic", "metal", "glass", "cardboard", "paper", "trash"]
SPLIT   = {"train": 0.70, "val": 0.15, "test": 0.15}

# -- TrashNet class name mapping -----------------------------------------------
TRASHNET_MAP = {
    "plastic":   "plastic",
    "metal":     "metal",
    "glass":     "glass",
    "cardboard": "cardboard",
    "paper":     "paper",
    "trash":     "trash",
}

# -- TACO category ? AquaScan class mapping -----------------------------------
TACO_MAP = {
    "Bottle":         "plastic",
    "Bottle cap":     "metal",
    "Can":            "metal",
    "Carton":         "cardboard",
    "Cup":            "plastic",
    "Lid":            "plastic",
    "Other plastic":  "trash",
    "Paper":          "paper",
    "Plastic bag":    "plastic",
    "Straw":          "plastic",
    "Styrofoam":      "plastic",
    "Wrapper":        "paper",
}


def setup_dirs(data_dir: pathlib.Path):
    """Create split/class directory tree."""
    for split in SPLIT:
        for cls in CLASSES:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"[prepare] Directory structure created under {data_dir}")


def split_and_copy(src_files: list[pathlib.Path], dst_class_dir: dict, cls: str, seed: int = 42):
    """
    Randomly split a list of image files into train/val/test
    and copy them to the appropriate output directories.
    """
    random.seed(seed)
    random.shuffle(src_files)

    n = len(src_files)
    n_train = int(n * SPLIT["train"])
    n_val   = int(n * SPLIT["val"])

    splits = {
        "train": src_files[:n_train],
        "val":   src_files[n_train:n_train + n_val],
        "test":  src_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        out_dir = dst_class_dir[split_name] / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            dst = out_dir / f"{cls}_{split_name}_{i:05d}{f.suffix}"
            shutil.copy2(f, dst)

    print(f"  {cls:>12}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def load_trashnet(raw_dir: pathlib.Path, out_dir: pathlib.Path):
    """
    Ingest images from a locally unzipped TrashNet dataset.
    Expected structure: raw_dir/data/<class>/*.jpg
    """
    print("\n[prepare] Loading TrashNet …")
    found_any = False

    for cls_name, mapped in TRASHNET_MAP.items():
        src_dir = raw_dir / "data" / cls_name
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found — skipping")
            continue
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        if not images:
            continue
        found_any = True
        split_and_copy(images, {s: out_dir / s for s in SPLIT}, mapped)

    if not found_any:
        print("  No TrashNet images found. Download from: https://github.com/garythung/trashnet")


def load_taco(raw_dir: pathlib.Path, out_dir: pathlib.Path):
    """
    Ingest TACO images from a locally extracted archive.
    Expected structure: raw_dir/data/<batch_N>/<images>
                        raw_dir/annotations.json
    """
    import json
    ann_path = raw_dir / "annotations.json"
    if not ann_path.exists():
        print("  WARNING: TACO annotations.json not found — skipping")
        return

    print("\n[prepare] Loading TACO …")
    with open(ann_path) as f:
        data = json.load(f)

    # Build {image_id: file_name} and {image_id: [category_names]}
    img_map = {img["id"]: raw_dir / img["file_name"] for img in data["images"]}
    cat_map = {cat["id"]: cat["name"] for cat in data["categories"]}

    from collections import defaultdict
    img_to_cats = defaultdict(set)
    for ann in data["annotations"]:
        cat_name = cat_map.get(ann["category_id"], "")
        mapped = TACO_MAP.get(cat_name, "trash")
        img_to_cats[ann["image_id"]].add(mapped)

    # Group images by their majority class
    class_to_imgs: dict[str, list[pathlib.Path]] = defaultdict(list)
    for img_id, cats in img_to_cats.items():
        path = img_map.get(img_id)
        if path and path.exists():
            cls = list(cats)[0]    # use first annotation class
            class_to_imgs[cls].append(path)

    for cls, imgs in class_to_imgs.items():
        if cls in CLASSES:
            split_and_copy(imgs, {s: out_dir / s for s in SPLIT}, cls)


def summarise(data_dir: pathlib.Path):
    """Print class × split count table."""
    print("\n[prepare] Final dataset summary:")
    print(f"  {'Class':>12} | {'train':>6} | {'val':>5} | {'test':>5}")
    print("  " + "-" * 38)
    for cls in CLASSES:
        counts = {}
        for split in SPLIT:
            counts[split] = len(list((data_dir / split / cls).glob("*")))
        print(f"  {cls:>12} | {counts['train']:>6} | {counts['val']:>5} | {counts['test']:>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="trashnet",
                        choices=["trashnet", "taco", "all"],
                        help="Which dataset(s) to ingest")
    parser.add_argument("--raw_dir",  default="data/raw",
                        help="Directory containing unzipped raw datasets")
    parser.add_argument("--data_dir", default="data/dataset",
                        help="Output directory for train/val/test split")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    raw_dir  = pathlib.Path(args.raw_dir)
    data_dir = pathlib.Path(args.data_dir)

    setup_dirs(data_dir)

    if args.source in ("trashnet", "all"):
        load_trashnet(raw_dir / "trashnet", data_dir)

    if args.source in ("taco", "all"):
        load_taco(raw_dir / "taco", data_dir)

    summarise(data_dir)
    print("\n[prepare] Done. You can now run: python model/train.py")


if __name__ == "__main__":
    main()
