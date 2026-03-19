"""
convert_csv_to_yolo.py
━━━━━━━━━━━━━━━━━━━━━━
Convert the kaggle_road_pothole_images CSV annotations to YOLO format.

CSV columns: image_id, num_potholes, x, y, w, h
  • (x, y) = top-left corner in pixels
  • (w, h) = box width/height in pixels

Images are in a deeply nested layout:
  Dataset 1 (Simplex)/Dataset 1 (Simplex)/Train data/Positive data/
  Dataset 1 (Simplex)/Dataset 1 (Simplex)/Train data/Negative data/
  Dataset 1 (Simplex)/Dataset 1 (Simplex)/Test data/

Negative images get empty label files so YOLOv8 can use them as background.
"""

from __future__ import annotations

import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "raw_datasets"
PROCESSED_DIR = BASE_DIR / "processed_data"

DATASET_NAME = "kaggle_road_pothole_images"
CSV_PATH = RAW_DIR / DATASET_NAME / "train_df.csv"

# Image search directories (positive + negative + test)
IMAGE_SEARCH_DIRS: list[Path] = [
    RAW_DIR / DATASET_NAME / "Dataset 1 (Simplex)" / "Dataset 1 (Simplex)" / "Train data" / "Positive data",
    RAW_DIR / DATASET_NAME / "Dataset 1 (Simplex)" / "Dataset 1 (Simplex)" / "Train data" / "Negative data",
    RAW_DIR / DATASET_NAME / "Dataset 1 (Simplex)" / "Dataset 1 (Simplex)" / "Test data",
]

YOLO_CLASS_ID = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
INCLUDE_NEGATIVES = True  # Keep negative images with empty labels

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Helpers ──────────────────────────────────


def build_image_index(search_dirs: list[Path]) -> dict[str, Path]:
    """
    Build a lookup: image_id (stem, case-insensitive) → full Path.

    Scans all search directories for image files.
    """
    index: dict[str, Path] = {}
    for d in search_dirs:
        if not d.is_dir():
            logger.warning("Image search directory not found: %s", d)
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                key = f.stem.upper()  # image_ids like G0010033
                if key not in index:
                    index[key] = f
    logger.info("Image index built: %d images found", len(index))
    return index


def parse_csv(csv_path: Path) -> dict[str, list[dict]]:
    """
    Parse the CSV annotation file.

    Returns
    -------
    dict mapping image_id → list of bbox dicts {x, y, w, h} (pixel coords).
    """
    annotations: dict[str, list[dict]] = defaultdict(list)
    seen_rows: set[tuple] = set()

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                image_id = row["image_id"].strip()
                x = int(row["x"])
                y = int(row["y"])
                w = int(row["w"])
                h = int(row["h"])
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping bad CSV row: %s", exc)
                continue

            # Deduplicate rows (the CSV has duplicated rows for some images)
            row_key = (image_id, x, y, w, h)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)

            if w <= 0 or h <= 0:
                logger.debug("Zero-area box for %s, skipping", image_id)
                continue

            annotations[image_id].append({"x": x, "y": y, "w": w, "h": h})

    logger.info("Parsed %d unique images from CSV with annotations", len(annotations))
    return annotations


def pixel_to_yolo(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
) -> Optional[tuple[float, float, float, float]]:
    """
    Convert pixel (x_topleft, y_topleft, w, h) to normalised YOLO
    (x_center, y_center, w, h).

    Returns None if the resulting box is invalid.
    """
    # Clamp to image boundaries
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    x1 = max(x, 0)
    y1 = max(y, 0)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + bw / 2.0) / img_w
    yc = (y1 + bh / 2.0) / img_h
    nw = bw / img_w
    nh = bh / img_h

    # Sanity
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
        return None

    return xc, yc, nw, nh


# ──────────────────────────── Main ─────────────────────────────────────


def run() -> None:
    """Convert kaggle_road_pothole_images CSV → YOLO labels."""
    logger.info("=" * 60)
    logger.info("CSV → YOLO Conversion (%s)", DATASET_NAME)
    logger.info("=" * 60)

    if not CSV_PATH.is_file():
        logger.error("CSV file not found: %s", CSV_PATH)
        return

    output_dir = PROCESSED_DIR / DATASET_NAME
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(IMAGE_SEARCH_DIRS)
    annotations = parse_csv(CSV_PATH)

    converted = 0
    skipped = 0
    negatives_added = 0

    # ── Process annotated (positive) images ──────────────────────────
    annotated_ids = set(annotations.keys())

    for image_id in tqdm(sorted(annotated_ids), desc="  Positive images", unit="img"):
        img_key = image_id.upper()
        if img_key not in image_index:
            logger.debug("Image not found for id=%s", image_id)
            skipped += 1
            continue

        img_path = image_index[img_key]
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image %s", img_path)
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        yolo_lines: list[str] = []
        for bbox in annotations[image_id]:
            result = pixel_to_yolo(bbox["x"], bbox["y"], bbox["w"], bbox["h"], img_w, img_h)
            if result is None:
                continue
            xc, yc, bw, bh = result
            yolo_lines.append(f"{YOLO_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            skipped += 1
            continue

        # Copy image
        out_img_path = out_images / f"{img_path.stem}.jpg"
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(img_path, out_img_path)
        else:
            cv2.imwrite(str(out_img_path), img)

        # Write label
        out_lbl_path = out_labels / f"{img_path.stem}.txt"
        out_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        converted += 1

    # ── Process negative images (empty labels) ───────────────────────
    if INCLUDE_NEGATIVES:
        for img_key, img_path in tqdm(
            sorted(image_index.items()),
            desc="  Negative images",
            unit="img",
        ):
            image_id_upper = img_path.stem.upper()
            # Skip if already processed as positive
            if image_id_upper in {k.upper() for k in annotated_ids}:
                continue

            # Check if this image comes from a "Negative" folder
            if "negative" not in str(img_path.parent).lower():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            out_img_path = out_images / f"{img_path.stem}.jpg"
            if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copy2(img_path, out_img_path)
            else:
                cv2.imwrite(str(out_img_path), img)

            # Empty label file for negative
            out_lbl_path = out_labels / f"{img_path.stem}.txt"
            out_lbl_path.write_text("", encoding="utf-8")
            negatives_added += 1

    logger.info(
        "CSV conversion done: converted=%d  skipped=%d  negatives=%d",
        converted,
        skipped,
        negatives_added,
    )


if __name__ == "__main__":
    run()
