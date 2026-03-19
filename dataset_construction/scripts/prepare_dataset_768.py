"""
prepare_dataset_768.py
━━━━━━━━━━━━━━━━━━━━━━
Letterbox-resize the entire YOLO-format pothole dataset to 768×768
and write the result into a NEW `dataset_768/` directory.

Key guarantees
──────────────
• Original `final_dataset/` is NEVER modified.
• Aspect ratio is preserved (letterbox padding, no distortion).
• YOLO bounding boxes are correctly recalculated after resize + pad.
• Label format remains:  class_id  x_center  y_center  width  height
  with all values normalised to [0, 1].
• Class IDs are untouched.
• Processes train, val, and test splits in a single run.

Usage
─────
    python prepare_dataset_768.py
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent          # dataset_construction/
SRC_DATASET = BASE_DIR / "final_dataset"                   # original dataset
DST_DATASET = BASE_DIR.parent / "dataset_768"              # output (project root)

TARGET_SIZE = 768                                          # square target
PAD_COLOR   = (114, 114, 114)                              # YOLO standard gray

SPLITS      = ["train", "val", "test"]                     # splits to process
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Core Logic ───────────────────────────────


def letterbox_resize(
    img: np.ndarray,
    target_size: int = TARGET_SIZE,
    pad_color: tuple[int, int, int] = PAD_COLOR,
) -> tuple[np.ndarray, float, int, int]:
    """
    Letterbox-resize an image to target_size × target_size.

    Returns
    -------
    canvas  : The padded, resized image.
    scale   : Scale factor applied to the original image.
    pad_x   : Horizontal padding added (left side).
    pad_y   : Vertical   padding added (top  side).
    """
    h, w = img.shape[:2]

    # Scale so the longest side fits exactly into target_size
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Choose interpolation: INTER_AREA for shrinking, INTER_LINEAR for enlarging
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Center the resized image on a padded canvas
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def adjust_yolo_labels(
    label_path: Path,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    target_size: int = TARGET_SIZE,
) -> list[str]:
    """
    Adjust YOLO normalised labels for the letterbox transformation.

    Maths
    ─────
    Original label is normalised w.r.t. (orig_w, orig_h).
    After letterboxing to target_size:
        pixel_x  = norm_x * orig_w * scale  +  pad_x
        norm_x'  = pixel_x / target_size

    Box width/height only scale (no offset):
        norm_w'  = norm_w * orig_w * scale / target_size
    """
    if not label_path.is_file():
        return []

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return [""]  # preserve empty-label files (background images)

    adjusted: list[str] = []

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            continue

        # Original normalised → pixel
        px_xc = xc * orig_w
        px_yc = yc * orig_h
        px_bw = bw * orig_w
        px_bh = bh * orig_h

        # Apply letterbox transformation
        new_px_xc = px_xc * scale + pad_x
        new_px_yc = px_yc * scale + pad_y
        new_px_bw = px_bw * scale
        new_px_bh = px_bh * scale

        # Re-normalise to target_size
        new_xc = new_px_xc / target_size
        new_yc = new_px_yc / target_size
        new_bw = new_px_bw / target_size
        new_bh = new_px_bh / target_size

        # Clamp to [0, 1]
        new_xc = max(0.0, min(1.0, new_xc))
        new_yc = max(0.0, min(1.0, new_yc))
        new_bw = max(0.0, min(1.0, new_bw))
        new_bh = max(0.0, min(1.0, new_bh))

        # Skip degenerate boxes
        if new_bw <= 0 or new_bh <= 0:
            continue

        adjusted.append(
            f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}"
        )

    return adjusted


# ──────────────────────────── Directory Setup ──────────────────────────


def create_output_dirs() -> None:
    """Create the dataset_768/ directory tree (images & labels per split)."""
    for split in SPLITS:
        (DST_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)
    logger.info("Output directory created: %s", DST_DATASET)


# ──────────────────────────── Processing ───────────────────────────────


def process_split(split: str) -> dict[str, int]:
    """
    Process one split (train / val / test).

    Returns a dict with counts: processed, skipped, errors.
    """
    src_img_dir = SRC_DATASET / "images" / split
    src_lbl_dir = SRC_DATASET / "labels" / split
    dst_img_dir = DST_DATASET / "images" / split
    dst_lbl_dir = DST_DATASET / "labels" / split

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    if not src_img_dir.is_dir():
        logger.warning("Source image directory missing, skipping split '%s': %s",
                        split, src_img_dir)
        return stats

    image_files = sorted(
        f for f in src_img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        logger.warning("No images found in '%s'", src_img_dir)
        return stats

    logger.info("Processing split %-5s — %d images", f"'{split}'", len(image_files))

    for img_path in tqdm(image_files, desc=f"  {split:>5}", unit="img", leave=True):
        # ── Read image ──────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: %s", img_path.name)
            stats["errors"] += 1
            continue

        orig_h, orig_w = img.shape[:2]

        # ── Letterbox resize ────────────────────────────────────────
        canvas, scale, pad_x, pad_y = letterbox_resize(img)

        # ── Save resized image (always .jpg) ────────────────────────
        dst_img_path = dst_img_dir / f"{img_path.stem}.jpg"
        if not cv2.imwrite(str(dst_img_path), canvas):
            logger.warning("Failed to write image: %s", dst_img_path.name)
            stats["errors"] += 1
            continue

        # ── Adjust and save labels ──────────────────────────────────
        src_lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"

        adjusted = adjust_yolo_labels(
            src_lbl_path, orig_w, orig_h, scale, pad_x, pad_y
        )

        if adjusted:
            text = "\n".join(adjusted)
            if text.strip():          # non-empty label
                text += "\n"
            dst_lbl_path.write_text(text, encoding="utf-8")
        else:
            # No source label exists — copy nothing (or create empty)
            dst_lbl_path.write_text("", encoding="utf-8")

        stats["processed"] += 1

    return stats


# ──────────────────────────── Data YAML ────────────────────────────────


def write_data_yaml() -> None:
    """Generate a data.yaml for the new dataset_768/ directory."""
    yaml_content = (
        f"path: {DST_DATASET.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n"
        f"- pothole\n"
    )
    yaml_path = DST_DATASET / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info("data.yaml written → %s", yaml_path)


# ──────────────────────────── Entry Point ──────────────────────────────


def main() -> None:
    """Resize the full dataset to 768×768 letterbox and save to dataset_768/."""
    t0 = time.perf_counter()

    logger.info("=" * 65)
    logger.info("  Pothole Dataset — Letterbox Resize to %d×%d", TARGET_SIZE, TARGET_SIZE)
    logger.info("=" * 65)
    logger.info("Source : %s", SRC_DATASET)
    logger.info("Output : %s", DST_DATASET)
    logger.info("")

    # Pre-flight checks
    if not SRC_DATASET.is_dir():
        logger.error("Source dataset not found: %s", SRC_DATASET)
        sys.exit(1)

    if DST_DATASET.exists():
        logger.warning("Output directory already exists — will merge/overwrite files.")

    create_output_dirs()

    # Process each split
    total = {"processed": 0, "skipped": 0, "errors": 0}
    for split in SPLITS:
        stats = process_split(split)
        for k in total:
            total[k] += stats[k]

    # Write data.yaml
    write_data_yaml()

    elapsed = time.perf_counter() - t0

    # Summary
    logger.info("")
    logger.info("=" * 65)
    logger.info("  DONE — Summary")
    logger.info("=" * 65)
    logger.info("  Processed : %d images", total["processed"])
    logger.info("  Skipped   : %d images", total["skipped"])
    logger.info("  Errors    : %d images", total["errors"])
    logger.info("  Elapsed   : %.1f seconds", elapsed)
    logger.info("  Output    : %s", DST_DATASET)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
