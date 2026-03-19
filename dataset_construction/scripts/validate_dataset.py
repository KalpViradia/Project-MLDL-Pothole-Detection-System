"""
validate_dataset.py
━━━━━━━━━━━━━━━━━━━
Phase 2: Comprehensive Dataset Validation for YOLOv8 Training.

Validates:
1. File structure (image-label pairs)
2. Label format (YOLO normalized 0-1, correct class ID)
3. Dataset statistics (images, boxes, negatives)
4. Bounding box size distribution (Small/Medium/Large)
5. Visual sanity checks (draw boxes on random samples)
6. Duplicate checks across splits

Outputs a JSON summary report and sample visualizations.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_ROOT = BASE_DIR / "final_dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
DATA_YAML = DATASET_ROOT / "data.yaml"

OUTPUT_DIR = BASE_DIR / "validation_outputs"
SAMPLE_DIRS = {
    "train": OUTPUT_DIR / "train_samples",
    "val": OUTPUT_DIR / "val_samples",
    "test": OUTPUT_DIR / "test_samples",
}

STATS_FILE = OUTPUT_DIR / "dataset_validation_report.json"

EXPECTED_CLASS_ID = 0
EXPECTED_NC = 1
EXPECTED_NAMES = ["pothole"]

# BBox Size Thresholds (pixels for 640x640 image)
# Small < 32px; Medium 32-96px; Large > 96px (COCO standard approximation)
IMG_SIZE = 640
AREA_SMALL = 32 * 32
AREA_MEDIUM = 96 * 96

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Helpers ──────────────────────────────────


class ValidationIssue(Exception):
    """Custom exception for validation errors (caught and logged)."""
    pass


def setup_directories():
    """Create output directories, potentially clearing old ones."""
    if OUTPUT_DIR.is_dir():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    for d in SAMPLE_DIRS.values():
        d.mkdir(parents=True)


def load_yaml(path: Path) -> dict:
    """Load data.yaml safely."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load data.yaml: %s", e)
        return {}


def yolo_to_pixel(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convert YOLO (xc, yc, w, h) to pixel (x1, y1, x2, y2)."""
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2


# ──────────────────────────── Validation Logic ─────────────────────────


def validate_structure(split: str) -> tuple[list[Path], list[Path], list[str]]:
    """
    Check image-label pairing for a given split.
    Returns: (image_paths, label_paths, issues)
    """
    img_split_dir = IMAGES_DIR / split
    lbl_split_dir = LABELS_DIR / split

    if not img_split_dir.is_dir():
        msg = f"Missing image directory: {img_split_dir}"
        logger.error(msg)
        return [], [], [msg]

    if not lbl_split_dir.is_dir():
        msg = f"Missing label directory: {lbl_split_dir}"
        logger.error(msg)
        return [], [], [msg]

    # Gather files
    images = sorted(
        f for f in img_split_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(
        f for f in lbl_split_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    )

    img_stems = {f.stem for f in images}
    lbl_stems = {f.stem for f in labels}

    issues = []

    # Check for missing labels
    missing_labels = img_stems - lbl_stems
    if missing_labels:
        issues.append(f"[{split}] {len(missing_labels)} images missing label files")
        for s in list(missing_labels)[:5]:
            logger.debug("  Missing label for image: %s", s)

    # Check for orphan labels
    orphan_labels = lbl_stems - img_stems
    if orphan_labels:
        issues.append(f"[{split}] {len(orphan_labels)} orphan label files (no image)")

    # Return matched pairs
    common_stems = img_stems & lbl_stems
    matched_images = [f for f in images if f.stem in common_stems]
    matched_labels = [lbl_split_dir / f"{f.stem}.txt" for f in matched_images]

    return matched_images, matched_labels, issues


def validate_labels(
    label_paths: list[Path],
) -> tuple[int, int, list[float], list[str], int, int]:
    """
    Validate content of label files.

    Returns:
        total_boxes
        empty_labels (negatives)
        box_areas (pixels, assuming 640x640)
        issues
        max_boxes_in_image
        min_boxes_in_image (excluding 0)
    """
    total_boxes = 0
    empty_labels = 0
    box_areas = []
    issues = []
    max_boxes = 0
    min_boxes = float("inf")

    for p in label_paths:
        try:
            content = p.read_text(encoding="utf-8").strip()
        except Exception as e:
            issues.append(f"Unreadable label {p.name}: {e}")
            continue

        if not content:
            empty_labels += 1
            max_boxes = max(max_boxes, 0)
            continue

        lines = content.splitlines()
        box_count = 0
        for line_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"Invalid format in {p.name}:{line_idx+1} (expected 5 val, got {len(parts)})")
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                issues.append(f"Non-numeric values in {p.name}:{line_idx+1}")
                continue

            if cls_id != EXPECTED_CLASS_ID:
                issues.append(f"Wrong class ID {cls_id} in {p.name}:{line_idx+1} (expected {EXPECTED_CLASS_ID})")

            # Check normalization
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                issues.append(f"Out-of-bounds coordinates in {p.name}:{line_idx+1}")

            # Collect metrics (assuming 640x640 for stats)
            pixel_w = w * IMG_SIZE
            pixel_h = h * IMG_SIZE
            area = pixel_w * pixel_h
            box_areas.append(area)
            box_count += 1

        total_boxes += box_count
        max_boxes = max(max_boxes, box_count)
        if box_count > 0:
            min_boxes = min(min_boxes, box_count)

    if min_boxes == float("inf"):
        min_boxes = 0

    return total_boxes, empty_labels, box_areas, issues, max_boxes, min_boxes


def check_duplicates_across_splits(all_images_map: dict[str, list[Path]]) -> list[str]:
    """
    Check if the same image name appears in multiple splits.
    (This is a basic name check; earlier perceptual hash handles content).
    """
    issues = []
    seen = defaultdict(list)
    for split, paths in all_images_map.items():
        for p in paths:
            seen[p.name].append(split)

    for name, splits in seen.items():
        if len(splits) > 1:
            issues.append(f"Image {name} appears in multiple splits: {splits}")

    return issues


def generate_visual_samples(
    split: str,
    images: list[Path],
    count: int,
    output_dir: Path,
):
    """Draw bounding boxes on random samples and save them."""
    if not images:
        return

    sample_images = random.sample(images, min(len(images), count))
    
    for img_path in sample_images:
        label_path = LABELS_DIR / split / f"{img_path.stem}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        if label_path.is_file():
            content = label_path.read_text(encoding="utf-8").strip()
            if content:
                for line in content.splitlines():
                    parts = line.split()
                    try:
                        xc, yc, bw, bh = map(float, parts[1:])
                        x1, y1, x2, y2 = yolo_to_pixel(xc, yc, bw, bh, w, h)
                        
                        # Draw box (Red)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Label size text
                        label = f"{int(bw*w)}x{int(bh*h)}"
                        cv2.putText(
                            img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                    except ValueError:
                        pass
        
        out_path = output_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), img)


# ──────────────────────────── Main Pipeline ────────────────────────────


def run() -> None:
    """Run comprehensive dataset validation."""
    logger.info("=" * 60)
    logger.info("STARTING DATASET VALIDATION")
    logger.info("=" * 60)
    
    setup_directories()

    # 1. Validate data.yaml
    logger.info("Validating data.yaml...")
    yaml_data = load_yaml(DATA_YAML)
    yaml_issues = []
    if yaml_data.get("nc") != EXPECTED_NC:
        yaml_issues.append(f"data.yaml nc={yaml_data.get('nc')} (expected {EXPECTED_NC})")
    if yaml_data.get("names") != EXPECTED_NAMES:
        yaml_issues.append(f"data.yaml names={yaml_data.get('names')} (expected {EXPECTED_NAMES})")
    
    if yaml_issues:
        logger.warning("YAML Issues: %s", yaml_issues)
    else:
        logger.info("  ✓ data.yaml is valid.")

    # 2. Process Splits
    splits = ["train", "val", "test"]
    all_stats = {}
    all_issues = yaml_issues
    all_images_map = {}
    
    total_imgs_all = 0
    total_boxes_all = 0
    all_areas = []

    for split in splits:
        logger.info("Processing split: %s", split)
        
        # Structure Check
        images, labels, struct_issues = validate_structure(split)
        all_issues.extend(struct_issues)
        all_images_map[split] = images
        
        logger.info("  %d images, %d matching labels", len(images), len(labels))
        
        # Label Content Check
        n_box, n_empty, areas, lbl_issues, max_b, min_b = validate_labels(labels)
        all_issues.extend(lbl_issues)
        
        # Stats per split
        n_img = len(images)
        avg_box = n_box / n_img if n_img > 0 else 0
        neg_pct = (n_empty / n_img * 100) if n_img > 0 else 0
        
        all_stats[split] = {
            "images": n_img,
            "boxes": n_box,
            "avg_boxes_per_image": round(avg_box, 2),
            "negative_images": n_empty,
            "negative_percentage": round(neg_pct, 2),
            "max_boxes": max_b,
            "min_boxes_non_empty": min_b if min_b != float("inf") else 0
        }
        
        total_imgs_all += n_img
        total_boxes_all += n_box
        all_areas.extend(areas)
        
        # Visual Sanity Check
        sample_count = {"train": 30, "val": 15, "test": 10}.get(split, 10)
        logger.info("  Generating %d visual samples...", sample_count)
        generate_visual_samples(split, images, sample_count, SAMPLE_DIRS[split])

    # 3. Duplicate Check
    logger.info("Checking for cross-split duplicates...")
    dup_issues = check_duplicates_across_splits(all_images_map)
    all_issues.extend(dup_issues)
    if not dup_issues:
        logger.info("  ✓ No cross-split name collisions found.")
    else:
        logger.warning("  ! Found %d cross-split duplicates", len(dup_issues))

    # 4. Box Size Analysis
    logger.info("Analyzing bounding box sizes...")
    n_small = sum(1 for a in all_areas if a < AREA_SMALL)
    n_medium = sum(1 for a in all_areas if AREA_SMALL <= a < AREA_MEDIUM)
    n_large = sum(1 for a in all_areas if a >= AREA_MEDIUM)
    n_total_area = len(all_areas)
    
    size_dist = {}
    if n_total_area > 0:
        size_dist = {
            "small": {
                "count": n_small,
                "percent": round(n_small / n_total_area * 100, 2),
                "desc": "< 32x32 px"
            },
            "medium": {
                "count": n_medium,
                "percent": round(n_medium / n_total_area * 100, 2),
                "desc": "32x32 - 96x96 px"
            },
            "large": {
                "count": n_large,
                "percent": round(n_large / n_total_area * 100, 2),
                "desc": "> 96x96 px"
            }
        }
        
    logger.info(
        "  S: %.1f%%  M: %.1f%%  L: %.1f%%", 
        size_dist.get("small", {}).get("percent", 0),
        size_dist.get("medium", {}).get("percent", 0),
        size_dist.get("large", {}).get("percent", 0)
    )

    # 5. Final Report
    report = {
        "status": "VALID" if not all_issues else "ISSUES_FOUND",
        "total_images": total_imgs_all,
        "total_boxes": total_boxes_all,
        "split_stats": all_stats,
        "box_size_distribution": size_dist,
        "issues": all_issues
    }
    
    with STATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("Report saved to: %s", STATS_FILE)
    logger.info("Visual checks in: %s", OUTPUT_DIR)
    
    if all_issues:
        logger.warning("Found %d issues! Check the report.", len(all_issues))
        # Print top 5 issues to console
        for i, issue in enumerate(all_issues[:5]):
            logger.warning("  - %s", issue)
    else:
        logger.info("SUCCESS: No issues found.")


if __name__ == "__main__":
    run()
