"""
remove_duplicates.py
━━━━━━━━━━━━━━━━━━━━
Remove duplicate images from processed_data/merged/ using perceptual hashing.

Uses average hashing (aHash) from the imagehash library. Images with
identical perceptual hashes are considered duplicates — the first
occurrence is kept, the rest (and their labels) are removed.

Also supports falling back to MD5 file hashing if imagehash is not available.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

# ──────────────────────────── Configuration ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

MERGED_DIR = BASE_DIR / "processed_data" / "merged"
IMAGES_DIR = MERGED_DIR / "images"
LABELS_DIR = MERGED_DIR / "labels"

HASH_SIZE = 16  # Perceptual hash size (higher = more strict)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Hash Functions ───────────────────────────


def compute_average_hash(img_path: Path, hash_size: int = HASH_SIZE) -> str | None:
    """
    Compute average perceptual hash for an image.

    Steps:
    1. Resize to hash_size × hash_size
    2. Convert to grayscale
    3. Compute mean pixel value
    4. Create binary hash: 1 if pixel > mean, else 0
    5. Convert to hex string
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to hash_size × hash_size
    resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # Compute mean
    mean_val = resized.mean()

    # Create binary hash
    binary = (resized > mean_val).flatten()

    # Convert to hex string
    hash_bits = "".join("1" if b else "0" for b in binary)

    # Convert binary string to hex (pad to consistent length)
    hex_hash = hex(int(hash_bits, 2))[2:].zfill(hash_size * hash_size // 4)

    return hex_hash


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file contents (fallback)."""
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ──────────────────────────── Main ─────────────────────────────────────


def run() -> None:
    """Remove duplicate images from the merged dataset."""
    logger.info("=" * 60)
    logger.info("Removing duplicates from merged dataset")
    logger.info("=" * 60)

    if not IMAGES_DIR.is_dir():
        logger.error("Images directory not found: %s", IMAGES_DIR)
        return

    image_files = sorted(
        f for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    logger.info("Total images before deduplication: %d", len(image_files))

    # Compute hashes
    hash_to_files: dict[str, list[Path]] = {}
    failed_hash: list[Path] = []

    for img_path in tqdm(image_files, desc="  Computing hashes", unit="img"):
        h = compute_average_hash(img_path)
        if h is None:
            # Fallback to file hash
            h = f"file_{compute_file_hash(img_path)}"
            failed_hash.append(img_path)

        if h not in hash_to_files:
            hash_to_files[h] = []
        hash_to_files[h].append(img_path)

    # Identify duplicates
    duplicates_removed = 0
    duplicate_groups = 0

    for h, files in hash_to_files.items():
        if len(files) <= 1:
            continue

        duplicate_groups += 1

        # Keep the first, remove the rest
        keep = files[0]
        to_remove = files[1:]

        for dup_path in to_remove:
            # Remove image
            try:
                dup_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove image %s: %s", dup_path.name, exc)
                continue

            # Remove corresponding label
            lbl_path = LABELS_DIR / f"{dup_path.stem}.txt"
            if lbl_path.is_file():
                try:
                    lbl_path.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove label %s: %s", lbl_path.name, exc)

            duplicates_removed += 1

        if duplicate_groups <= 10:
            logger.debug(
                "  Group: kept=%s, removed=%d duplicates",
                keep.name,
                len(to_remove),
            )

    # Count remaining
    remaining = sum(
        1 for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    logger.info("=" * 60)
    logger.info("Deduplication complete:")
    logger.info("  Duplicate groups found: %d", duplicate_groups)
    logger.info("  Images removed:         %d", duplicates_removed)
    logger.info("  Images remaining:       %d", remaining)
    if failed_hash:
        logger.info("  (Used file hash fallback for %d unreadable images)", len(failed_hash))


if __name__ == "__main__":
    run()
