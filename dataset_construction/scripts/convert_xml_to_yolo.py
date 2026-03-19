"""
convert_xml_to_yolo.py
━━━━━━━━━━━━━━━━━━━━━━
Convert Pascal VOC XML annotations to YOLO format.

Handles two dataset layouts:
  • kaggle_annotated_potholes  – XML + JPG mixed in one folder
  • kaggle_pothole_665         – separate annotations/ and images/ folders

Only keeps objects whose name matches the target class (default: 'pothole').
All kept objects are remapped to class_id = 0.
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "raw_datasets"
PROCESSED_DIR = BASE_DIR / "processed_data"

# Dataset-specific paths
DATASETS: dict[str, dict] = {
    "kaggle_annotated_potholes": {
        "xml_dir": RAW_DIR / "kaggle_annotated_potholes" / "annotated-images",
        "img_dir": RAW_DIR / "kaggle_annotated_potholes" / "annotated-images",
        "layout": "mixed",  # XML and images in same folder
    },
    "kaggle_pothole_665": {
        "xml_dir": RAW_DIR / "kaggle_pothole_665" / "annotations",
        "img_dir": RAW_DIR / "kaggle_pothole_665" / "images",
        "layout": "separate",  # XML in one folder, images in another
    },
}

TARGET_CLASS = "pothole"
YOLO_CLASS_ID = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Core Logic ───────────────────────────────


def parse_voc_xml(xml_path: Path) -> Optional[dict]:
    """
    Parse a Pascal VOC XML file and return image metadata + bounding boxes.

    Returns
    -------
    dict with keys: filename, width, height, objects (list of bbox dicts)
    None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        logger.warning("Failed to parse XML %s: %s", xml_path.name, exc)
        return None

    filename_el = root.find("filename")
    size_el = root.find("size")
    if filename_el is None or size_el is None:
        logger.warning("Missing <filename> or <size> in %s", xml_path.name)
        return None

    filename = filename_el.text.strip() if filename_el.text else ""

    width_el = size_el.find("width")
    height_el = size_el.find("height")
    if width_el is None or height_el is None:
        logger.warning("Missing <width> or <height> in %s", xml_path.name)
        return None

    try:
        width = int(width_el.text)
        height = int(height_el.text)
    except (ValueError, TypeError):
        logger.warning("Non-integer dimensions in %s", xml_path.name)
        return None

    if width <= 0 or height <= 0:
        logger.warning("Invalid dimensions (%d×%d) in %s", width, height, xml_path.name)
        return None

    objects: list[dict] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue

        obj_name = name_el.text.strip().lower()
        if obj_name != TARGET_CLASS.lower():
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (AttributeError, ValueError, TypeError):
            logger.warning("Bad bndbox values in %s", xml_path.name)
            continue

        # Clamp to image boundaries
        xmin = max(0.0, min(xmin, width))
        ymin = max(0.0, min(ymin, height))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))

        # Validate
        if xmax <= xmin or ymax <= ymin:
            logger.debug("Degenerate box skipped in %s", xml_path.name)
            continue

        objects.append({
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def voc_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert VOC bbox (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, w, h) normalised."""
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def find_image_for_xml(
    xml_path: Path,
    img_dir: Path,
    filename_hint: str,
) -> Optional[Path]:
    """
    Locate the image file corresponding to an XML annotation.

    Tries the filename from XML metadata first, then falls back to
    matching the XML stem with common image extensions.
    """
    # Try filename from XML
    if filename_hint:
        candidate = img_dir / filename_hint
        if candidate.is_file():
            return candidate

    # Fallback: match by stem
    stem = xml_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    return None


def convert_dataset(
    dataset_name: str,
    xml_dir: Path,
    img_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """
    Convert a single VOC-annotated dataset to YOLO format.

    Parameters
    ----------
    dataset_name : str
        Human-readable name (used in logging).
    xml_dir : Path
        Directory containing .xml annotation files.
    img_dir : Path
        Directory containing image files.
    output_dir : Path
        Where to write converted images + labels.

    Returns
    -------
    dict with conversion statistics.
    """
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        logger.warning("No XML files found in %s", xml_dir)
        return {"total": 0, "converted": 0, "skipped": 0, "no_image": 0}

    stats = {"total": len(xml_files), "converted": 0, "skipped": 0, "no_image": 0}

    logger.info("Converting dataset '%s': %d XML files found", dataset_name, len(xml_files))

    for xml_path in tqdm(xml_files, desc=f"  {dataset_name}", unit="file"):
        parsed = parse_voc_xml(xml_path)
        if parsed is None:
            stats["skipped"] += 1
            continue

        if not parsed["objects"]:
            stats["skipped"] += 1
            continue

        img_path = find_image_for_xml(xml_path, img_dir, parsed["filename"])
        if img_path is None:
            stats["no_image"] += 1
            logger.debug("No image found for %s", xml_path.name)
            continue

        # Verify image is readable and get actual dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            logger.warning("Cannot read image %s", img_path.name)
            continue

        actual_h, actual_w = img.shape[:2]

        # Use actual image dimensions (more reliable than XML metadata)
        img_w = actual_w
        img_h = actual_h

        # Build YOLO labels
        yolo_lines: list[str] = []
        for obj in parsed["objects"]:
            # Re-clamp against actual dimensions
            xmin = max(0.0, min(obj["xmin"], img_w))
            ymin = max(0.0, min(obj["ymin"], img_h))
            xmax = max(0.0, min(obj["xmax"], img_w))
            ymax = max(0.0, min(obj["ymax"], img_h))

            if xmax <= xmin or ymax <= ymin:
                continue

            xc, yc, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

            # Final sanity check
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                logger.debug("Out-of-range YOLO coords for %s, skipping box", xml_path.name)
                continue

            yolo_lines.append(f"{YOLO_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            stats["skipped"] += 1
            continue

        # Copy image as .jpg
        out_img_name = f"{img_path.stem}.jpg"
        out_img_path = out_images / out_img_name
        if img_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(img_path, out_img_path)
        else:
            cv2.imwrite(str(out_img_path), img)

        # Write label
        out_lbl_path = out_labels / f"{img_path.stem}.txt"
        out_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        stats["converted"] += 1

    return stats


# ──────────────────────────── Entry Point ──────────────────────────────


def run() -> None:
    """Convert all configured XML datasets to YOLO format."""
    logger.info("=" * 60)
    logger.info("XML → YOLO Conversion")
    logger.info("=" * 60)

    total_converted = 0

    for ds_name, ds_cfg in DATASETS.items():
        xml_dir: Path = ds_cfg["xml_dir"]
        img_dir: Path = ds_cfg["img_dir"]

        if not xml_dir.is_dir():
            logger.error("XML directory not found: %s", xml_dir)
            continue

        if not img_dir.is_dir():
            logger.error("Image directory not found: %s", img_dir)
            continue

        output_dir = PROCESSED_DIR / ds_name
        stats = convert_dataset(ds_name, xml_dir, img_dir, output_dir)

        logger.info(
            "  %s → converted=%d  skipped=%d  no_image=%d  total=%d",
            ds_name,
            stats["converted"],
            stats["skipped"],
            stats["no_image"],
            stats["total"],
        )
        total_converted += stats["converted"]

    logger.info("Total images converted from XML: %d", total_converted)


if __name__ == "__main__":
    run()
