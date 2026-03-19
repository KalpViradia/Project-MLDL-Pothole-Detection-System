"""
dataset_splitter.py
━━━━━━━━━━━━━━━━━━━
Split the deduplicated merged dataset into train / val / test
and produce the final YOLO-ready folder structure:

  final_dataset/
    images/
      train/  val/  test/
    labels/
      train/  val/  test/
    data.yaml

Uses stratified splitting based on the number of bounding boxes
per image (bins: 0, 1-3, 4+) to preserve class / density distribution.
"""

from __future__ import annotations

import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

MERGED_DIR = BASE_DIR / "processed_data" / "merged"
FINAL_DIR = BASE_DIR / "final_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# YOLO dataset metadata
NUM_CLASSES = 1
CLASS_NAMES = ["pothole"]

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Helpers ──────────────────────────────────


def count_boxes(label_path: Path) -> int:
    """Count the number of bounding boxes in a YOLO label file."""
    if not label_path.is_file():
        return 0

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return 0

    return sum(1 for line in content.splitlines() if line.strip())


def get_stratum(box_count: int) -> str:
    """Assign an image to a stratum based on box count."""
    if box_count == 0:
        return "0_boxes"
    elif box_count <= 3:
        return "1-3_boxes"
    else:
        return "4+_boxes"


def stratified_split(
    items: list[tuple[Path, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Perform stratified splitting of items.

    Parameters
    ----------
    items : list of (image_path, stratum_label)
    train_ratio, val_ratio : split ratios (test = 1 - train - val)
    seed : random seed

    Returns
    -------
    (train_paths, val_paths, test_paths) — each a list of image Paths
    """
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[str, list[Path]] = {}
    for img_path, stratum in items:
        if stratum not in strata:
            strata[stratum] = []
        strata[stratum].append(img_path)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    for stratum_name, paths in sorted(strata.items()):
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 in each split if possible
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            # Adjust if over
            total = n_train + n_val + n_test
            if total > n:
                n_train = n - n_val - n_test

        n_test = n - n_train - n_val

        train.extend(paths[:n_train])
        val.extend(paths[n_train : n_train + n_val])
        test.extend(paths[n_train + n_val :])

        logger.info(
            "  Stratum '%s': %d total → train=%d  val=%d  test=%d",
            stratum_name,
            n,
            n_train,
            n_val,
            n_test,
        )

    return train, val, test


# ──────────────────────────── Main ─────────────────────────────────────


def run() -> None:
    """Split merged dataset into train/val/test and create data.yaml."""
    logger.info("=" * 60)
    logger.info("Dataset Splitter → %.0f%% / %.0f%% / %.0f%%",
                TRAIN_RATIO * 100, VAL_RATIO * 100, TEST_RATIO * 100)
    logger.info("=" * 60)

    images_dir = MERGED_DIR / "images"
    labels_dir = MERGED_DIR / "labels"

    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        return

    # Collect all images with their strata
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info("Total images for splitting: %d", len(image_files))

    items: list[tuple[Path, str]] = []
    stratum_counts: Counter = Counter()

    for img_path in image_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        n_boxes = count_boxes(lbl_path)
        stratum = get_stratum(n_boxes)
        items.append((img_path, stratum))
        stratum_counts[stratum] += 1

    logger.info("Stratum distribution: %s", dict(stratum_counts))

    # Split
    train_paths, val_paths, test_paths = stratified_split(
        items, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    logger.info(
        "Split result: train=%d  val=%d  test=%d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    # Create final directory structure
    splits = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }

    for split_name, paths in splits.items():
        dst_imgs = FINAL_DIR / "images" / split_name
        dst_lbls = FINAL_DIR / "labels" / split_name
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(paths, desc=f"  → {split_name}", unit="img"):
            # Copy image
            shutil.copy2(img_path, dst_imgs / img_path.name)

            # Copy label
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            dst_lbl = dst_lbls / f"{img_path.stem}.txt"
            if lbl_path.is_file():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

    # Generate data.yaml
    data_yaml = {
        "path": str(FINAL_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    yaml_path = FINAL_DIR / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.dump(data_yaml, fh, default_flow_style=False, sort_keys=False)

    logger.info("=" * 60)
    logger.info("Final dataset created at: %s", FINAL_DIR)
    logger.info("data.yaml written to: %s", yaml_path)
    logger.info(
        "Summary: train=%d  val=%d  test=%d  total=%d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
        len(train_paths) + len(val_paths) + len(test_paths),
    )


if __name__ == "__main__":
    run()
