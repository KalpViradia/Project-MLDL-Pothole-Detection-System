"""
resize_and_standardize.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Letterbox-resize all images in processed_data/merged/ to 640×640.

Letterboxing preserves aspect ratio by padding with gray (114, 114, 114)
— the same pad value YOLOv8 uses by default.

Bounding box coordinates in the label files are adjusted to account
for the scaling and padding so they remain accurate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

MERGED_DIR = BASE_DIR / "processed_data" / "merged"
IMAGES_DIR = MERGED_DIR / "images"
LABELS_DIR = MERGED_DIR / "labels"

TARGET_SIZE = 640  # Square target
PAD_COLOR = (114, 114, 114)  # YOLO standard gray padding

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    target_size : int
        Desired output dimension (square).
    pad_color : tuple
        BGR color for padding.

    Returns
    -------
    resized : np.ndarray
        The letterboxed image (target_size × target_size).
    scale : float
        Scale factor applied to the original image.
    pad_x : int
        Horizontal padding added (one side).
    pad_y : int
        Vertical padding added (one side).
    """
    h, w = img.shape[:2]

    # Compute scale (fit the longer side to target_size)
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    # Create canvas and place resized image
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
    Adjust YOLO normalised labels for letterbox transformation.

    The original labels are normalised w.r.t. original image dimensions.
    After letterboxing:
      new_pixel_x = (orig_norm_x * orig_w) * scale + pad_x
      new_norm_x  = new_pixel_x / target_size

    Same for y.
    Box width/height just scale (no offset needed):
      new_norm_w = (orig_norm_w * orig_w * scale) / target_size
    """
    adjusted: list[str] = []

    if not label_path.is_file():
        return adjusted

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return [""]  # Preserve empty labels

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except (ValueError, IndexError):
            continue

        # Convert from original normalised → original pixel
        px_xc = xc * orig_w
        px_yc = yc * orig_h
        px_bw = bw * orig_w
        px_bh = bh * orig_h

        # Apply letterbox transform
        new_px_xc = px_xc * scale + pad_x
        new_px_yc = px_yc * scale + pad_y
        new_px_bw = px_bw * scale
        new_px_bh = px_bh * scale

        # Normalise to target size
        new_xc = new_px_xc / target_size
        new_yc = new_px_yc / target_size
        new_bw = new_px_bw / target_size
        new_bh = new_px_bh / target_size

        # Clamp
        new_xc = max(0.0, min(1.0, new_xc))
        new_yc = max(0.0, min(1.0, new_yc))
        new_bw = max(0.0, min(1.0, new_bw))
        new_bh = max(0.0, min(1.0, new_bh))

        if new_bw <= 0 or new_bh <= 0:
            continue

        adjusted.append(
            f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}"
        )

    return adjusted


# ──────────────────────────── Entry Point ──────────────────────────────


def run() -> None:
    """Letterbox-resize all images to 640×640 and adjust labels."""
    logger.info("=" * 60)
    logger.info("Resize & Standardize → %d×%d letterbox", TARGET_SIZE, TARGET_SIZE)
    logger.info("=" * 60)

    if not IMAGES_DIR.is_dir():
        logger.error("Images directory not found: %s", IMAGES_DIR)
        return

    image_files = sorted(
        f for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    logger.info("Images to resize: %d", len(image_files))

    resized_count = 0
    already_ok = 0
    errors = 0

    for img_path in tqdm(image_files, desc="  Resizing", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s, skipping", img_path.name)
            errors += 1
            continue

        orig_h, orig_w = img.shape[:2]

        # Skip if already correct size
        if orig_h == TARGET_SIZE and orig_w == TARGET_SIZE:
            already_ok += 1
            continue

        # Letterbox resize
        canvas, scale, pad_x, pad_y = letterbox_resize(img)

        # Adjust labels
        lbl_path = LABELS_DIR / f"{img_path.stem}.txt"
        adjusted_lines = adjust_yolo_labels(
            lbl_path, orig_w, orig_h, scale, pad_x, pad_y
        )

        # Overwrite image (as .jpg)
        out_path = img_path.with_suffix(".jpg")
        cv2.imwrite(str(out_path), canvas)

        # If original was not .jpg, remove original
        if img_path.suffix.lower() not in {".jpg", ".jpeg"} and img_path != out_path:
            try:
                img_path.unlink()
            except OSError:
                pass

        # Overwrite label
        if adjusted_lines:
            text = "\n".join(adjusted_lines)
            # Don't add trailing newline if it's an empty label marker
            if text.strip():
                text += "\n"
            lbl_path.write_text(text, encoding="utf-8")

        resized_count += 1

    logger.info("=" * 60)
    logger.info("Resize complete:")
    logger.info("  Resized:    %d", resized_count)
    logger.info("  Already OK: %d", already_ok)
    logger.info("  Errors:     %d", errors)


if __name__ == "__main__":
    run()
