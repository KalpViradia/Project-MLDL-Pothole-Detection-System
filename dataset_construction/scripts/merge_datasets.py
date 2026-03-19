"""
merge_datasets.py
━━━━━━━━━━━━━━━━━
Merge all converted datasets from processed_data/<name>/ into
processed_data/merged/.

Each image is renamed to <datasetname>_<index>.jpg to avoid collisions.
A JSON manifest is written for provenance tracking.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "processed_data"
MERGED_DIR = PROCESSED_DIR / "merged"

# Datasets to merge (order doesn't matter — all get unique names)
DATASET_NAMES: list[str] = [
    "kaggle_annotated_potholes",
    "kaggle_pothole_665",
    "kaggle_road_pothole_images",
    "kaggle_yolov8_potholes",
    "rdd2022",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Helpers ──────────────────────────────────


def copy_yolo_dataset(
    dataset_name: str,
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    manifest: list[dict],
) -> int:
    """
    Copy images + labels from a single dataset into the merged directory.

    Renames files to <datasetname>_<index>.jpg.

    Returns the number of images copied.
    """
    if not src_images_dir.is_dir():
        logger.warning("Images dir not found for '%s': %s", dataset_name, src_images_dir)
        return 0

    image_files = sorted(
        f for f in src_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.warning("No images found for '%s'", dataset_name)
        return 0

    copied = 0
    for idx, img_path in enumerate(
        tqdm(image_files, desc=f"  {dataset_name}", unit="img")
    ):
        new_stem = f"{dataset_name}_{idx:05d}"
        new_img_name = f"{new_stem}.jpg"
        new_lbl_name = f"{new_stem}.txt"

        # Copy / convert image
        dst_img = dst_images_dir / new_img_name
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(img_path, dst_img)
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Cannot read %s, skipping", img_path)
                continue
            cv2.imwrite(str(dst_img), img)

        # Copy label (if exists)
        src_lbl = src_labels_dir / f"{img_path.stem}.txt"
        dst_lbl = dst_labels_dir / new_lbl_name
        if src_lbl.is_file():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # Empty label (negative image or missing)
            dst_lbl.write_text("", encoding="utf-8")

        manifest.append({
            "new_name": new_img_name,
            "original_name": img_path.name,
            "source_dataset": dataset_name,
            "source_path": str(img_path),
        })
        copied += 1

    return copied


# ──────────────────────────── Entry Point ──────────────────────────────


def run() -> None:
    """Merge all processed datasets into a single merged directory."""
    logger.info("=" * 60)
    logger.info("Merging all datasets → processed_data/merged/")
    logger.info("=" * 60)

    dst_images = MERGED_DIR / "images"
    dst_labels = MERGED_DIR / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    total_copied = 0

    for ds_name in DATASET_NAMES:
        ds_dir = PROCESSED_DIR / ds_name
        src_images = ds_dir / "images"
        src_labels = ds_dir / "labels"

        if not ds_dir.is_dir():
            logger.warning("Dataset '%s' not found in processed_data/, skipping", ds_name)
            continue

        count = copy_yolo_dataset(
            ds_name,
            src_images,
            src_labels,
            dst_images,
            dst_labels,
            manifest,
        )
        logger.info("  %s → %d images merged", ds_name, count)
        total_copied += count

    # Write manifest
    manifest_path = MERGED_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("Merge complete: %d total images in merged/", total_copied)
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    run()
