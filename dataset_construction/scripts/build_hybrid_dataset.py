"""
build_hybrid_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━
Main pipeline orchestrator for building the hybrid pothole detection dataset.

Pipeline steps:
  1. Convert XML datasets (kaggle_annotated_potholes, kaggle_pothole_665)
  2. Convert CSV dataset (kaggle_road_pothole_images)
  3. Filter RDD2022 to pothole-only
  4. Copy already-YOLO dataset (kaggle_yolov8_potholes)
  5. Merge all into processed_data/merged/
  6. Remove duplicates (perceptual hashing)
  7. Resize and standardize to 640×640 (letterbox)
  8. Stratified split into final_dataset/ (70/20/10)

Usage:
    python scripts/build_hybrid_dataset.py
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "raw_datasets"
PROCESSED_DIR = BASE_DIR / "processed_data"
FINAL_DIR = BASE_DIR / "final_dataset"
MERGED_DIR = PROCESSED_DIR / "merged"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Step Helpers ─────────────────────────────


def timed_step(step_name: str):
    """Decorator/context-manager that logs step timing."""

    class TimedContext:
        def __enter__(self):
            logger.info("")
            logger.info("━" * 60)
            logger.info("STEP: %s", step_name)
            logger.info("━" * 60)
            self.start = time.time()
            return self

        def __exit__(self, *_):
            elapsed = time.time() - self.start
            logger.info("  ✓ %s completed in %.1f seconds", step_name, elapsed)

    return TimedContext()


def copy_yolov8_potholes() -> None:
    """
    Copy the kaggle_yolov8_potholes dataset (already YOLO-format)
    to processed_data/kaggle_yolov8_potholes/.

    This dataset already has class 0 = pothole, so we just copy as-is.
    """
    ds_name = "kaggle_yolov8_potholes"
    src_root = RAW_DIR / ds_name
    dst_dir = PROCESSED_DIR / ds_name
    dst_images = dst_dir / "images"
    dst_labels = dst_dir / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for split in ["train", "valid"]:
        src_imgs = src_root / split / "images"
        src_lbls = src_root / split / "labels"

        if not src_imgs.is_dir():
            logger.warning("  %s/images not found, skipping", split)
            continue

        img_files = sorted(
            f for f in src_imgs.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        for img_path in tqdm(img_files, desc=f"  yolov8_potholes/{split}", unit="img"):
            # Copy image
            dst_img = dst_images / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy label
            lbl_path = src_lbls / f"{img_path.stem}.txt"
            dst_lbl = dst_labels / f"{img_path.stem}.txt"
            if lbl_path.is_file():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

            total_copied += 1

    logger.info("  kaggle_yolov8_potholes → %d images copied", total_copied)


def clean_output_dirs() -> None:
    """Remove existing output directories for a clean run."""
    for d in [MERGED_DIR, FINAL_DIR]:
        if d.is_dir():
            logger.info("  Cleaning %s", d)
            shutil.rmtree(d)


def print_final_stats() -> None:
    """Print statistics about the final dataset."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("=" * 60)

    for split in ["train", "val", "test"]:
        img_dir = FINAL_DIR / "images" / split
        lbl_dir = FINAL_DIR / "labels" / split

        n_imgs = 0
        n_lbls = 0
        total_boxes = 0
        empty_labels = 0

        if img_dir.is_dir():
            n_imgs = sum(
                1 for f in img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

        if lbl_dir.is_dir():
            for lbl in lbl_dir.iterdir():
                if lbl.is_file() and lbl.suffix == ".txt":
                    n_lbls += 1
                    content = lbl.read_text(encoding="utf-8").strip()
                    if not content:
                        empty_labels += 1
                    else:
                        total_boxes += len(content.splitlines())

        logger.info(
            "  %5s → images=%d  labels=%d  boxes=%d  empty=%d",
            split,
            n_imgs,
            n_lbls,
            total_boxes,
            empty_labels,
        )

    yaml_path = FINAL_DIR / "data.yaml"
    if yaml_path.is_file():
        logger.info("  data.yaml: %s", yaml_path)
        logger.info("  Contents:")
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            logger.info("    %s", line)


# ──────────────────────────── Main Pipeline ────────────────────────────


def main() -> None:
    """Run the full hybrid dataset build pipeline."""
    overall_start = time.time()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  HYBRID POTHOLE DETECTION DATASET BUILDER                ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info("Base directory: %s", BASE_DIR)
    logger.info("Raw data:       %s", RAW_DIR)
    logger.info("Processed:      %s", PROCESSED_DIR)
    logger.info("Final output:   %s", FINAL_DIR)

    # Import pipeline modules
    from convert_xml_to_yolo import run as run_xml_to_yolo
    from convert_csv_to_yolo import run as run_csv_to_yolo
    from convert_rdd2022_to_yolo import run as run_rdd2022_to_yolo
    from merge_datasets import run as run_merge
    from remove_duplicates import run as run_dedup
    from resize_and_standardize import run as run_resize
    from dataset_splitter import run as run_split

    # ── Step 0: Clean ────────────────────────────────────────────────
    with timed_step("Clean output directories"):
        clean_output_dirs()

    # ── Step 1: Convert XML datasets ─────────────────────────────────
    with timed_step("Convert XML → YOLO"):
        run_xml_to_yolo()

    # ── Step 2: Convert CSV dataset ──────────────────────────────────
    with timed_step("Convert CSV → YOLO"):
        run_csv_to_yolo()

    # ── Step 3: Filter RDD2022 ───────────────────────────────────────
    with timed_step("Filter RDD2022 → Pothole-only"):
        run_rdd2022_to_yolo()

    # ── Step 4: Copy already-YOLO dataset ────────────────────────────
    with timed_step("Copy kaggle_yolov8_potholes"):
        copy_yolov8_potholes()

    # ── Step 5: Merge all ────────────────────────────────────────────
    with timed_step("Merge all datasets"):
        run_merge()

    # ── Step 6: Remove duplicates ────────────────────────────────────
    with timed_step("Remove duplicates"):
        run_dedup()

    # ── Step 7: Resize and standardize ───────────────────────────────
    with timed_step("Resize & standardize to 640×640"):
        run_resize()

    # ── Step 8: Split ────────────────────────────────────────────────
    with timed_step("Stratified split (70/20/10)"):
        run_split()

    # ── Done ─────────────────────────────────────────────────────────
    print_final_stats()

    total_time = time.time() - overall_start
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  PIPELINE COMPLETE — Total time: %6.1f seconds           ║", total_time)
    logger.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
