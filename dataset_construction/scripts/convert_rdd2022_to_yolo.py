"""
convert_rdd2022_to_yolo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Filter RDD2022 YOLO labels to keep only pothole-class annotations.

RDD2022 uses multi-class labels. By default we assume the pothole class
is index 0 (D00 — longitudinal cracks are class 0 in the original, but
the specific mapping depends on how the dataset was prepared).

You can configure POTHOLE_CLASS_IDS to include multiple source class IDs
if needed (e.g., if potholes are class 2).

All kept annotations are remapped to class_id = 0.
Images without any pothole annotation are skipped.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "raw_datasets"
PROCESSED_DIR = BASE_DIR / "processed_data"

DATASET_NAME = "rdd2022"
RDD_ROOT = RAW_DIR / DATASET_NAME / "RDD_SPLIT"

# Source class IDs that represent potholes.
# RDD2022 typically: 0=D00, 1=D10, 2=D20, 3=D40
# D00 = longitudinal crack; D40 = pothole in some mappings.
# Adjust these IDs based on your specific dataset version.
POTHOLE_CLASS_IDS: set[int] = {0}

OUTPUT_CLASS_ID = 0  # All kept annotations become this class

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Core Logic ───────────────────────────────


def filter_label_file(label_path: Path) -> list[str]:
    """
    Read a YOLO label file and return only lines whose class_id
    is in POTHOLE_CLASS_IDS, remapped to OUTPUT_CLASS_ID.
    """
    filtered: list[str] = []
    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Cannot read label %s: %s", label_path.name, exc)
        return filtered

    if not content:
        return filtered

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(parts[0])
        except ValueError:
            continue

        if cls_id not in POTHOLE_CLASS_IDS:
            continue

        # Validate coordinate values
        try:
            coords = [float(p) for p in parts[1:5]]
        except ValueError:
            continue

        # Check normalised range
        if any(c < 0 or c > 1 for c in coords):
            logger.debug("Out-of-range coords in %s: %s", label_path.name, coords)
            continue

        # Check non-zero dimensions
        if coords[2] <= 0 or coords[3] <= 0:
            continue

        filtered.append(
            f"{OUTPUT_CLASS_ID} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
        )

    return filtered


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    """Locate the image file corresponding to a label file."""
    stem = label_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def process_split(split_name: str, split_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Process one split (train/val/test) of the RDD2022 dataset.

    Returns stats dict.
    """
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "skipped_no_pothole": 0, "skipped_no_image": 0}

    if not labels_dir.is_dir():
        logger.warning("Labels dir not found: %s", labels_dir)
        return stats

    label_files = sorted(labels_dir.glob("*.txt"))
    stats["total"] = len(label_files)

    if not label_files:
        logger.warning("No label files in %s", labels_dir)
        return stats

    logger.info("Processing RDD2022 split '%s': %d label files", split_name, len(label_files))

    for lbl_path in tqdm(label_files, desc=f"  rdd2022/{split_name}", unit="file"):
        filtered_lines = filter_label_file(lbl_path)

        if not filtered_lines:
            stats["skipped_no_pothole"] += 1
            continue

        img_path = find_image_for_label(lbl_path, images_dir)
        if img_path is None:
            stats["skipped_no_image"] += 1
            continue

        # Verify image readability
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped_no_image"] += 1
            logger.debug("Cannot read image %s", img_path.name)
            continue

        # Copy image
        out_img_path = out_images / f"{img_path.stem}.jpg"
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(img_path, out_img_path)
        else:
            cv2.imwrite(str(out_img_path), img)

        # Write filtered label
        out_lbl_path = out_labels / f"{img_path.stem}.txt"
        out_lbl_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

        stats["kept"] += 1

    return stats


# ──────────────────────────── Entry Point ──────────────────────────────


def run() -> None:
    """Filter RDD2022 to pothole-only and copy to processed_data/."""
    logger.info("=" * 60)
    logger.info("RDD2022 → YOLO (Pothole-only) Conversion")
    logger.info("Pothole source class IDs: %s", POTHOLE_CLASS_IDS)
    logger.info("=" * 60)

    if not RDD_ROOT.is_dir():
        logger.error("RDD2022 root not found: %s", RDD_ROOT)
        return

    output_dir = PROCESSED_DIR / DATASET_NAME
    total_kept = 0

    for split_name in ["train", "val", "test"]:
        split_dir = RDD_ROOT / split_name
        if not split_dir.is_dir():
            logger.warning("Split '%s' not found, skipping", split_name)
            continue

        stats = process_split(split_name, split_dir, output_dir)

        logger.info(
            "  %s → kept=%d  no_pothole=%d  no_image=%d  total=%d",
            split_name,
            stats["kept"],
            stats["skipped_no_pothole"],
            stats["skipped_no_image"],
            stats["total"],
        )
        total_kept += stats["kept"]

    logger.info("Total RDD2022 images kept (pothole-only): %d", total_kept)


if __name__ == "__main__":
    run()
