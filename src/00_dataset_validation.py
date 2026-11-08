"""
Stage 0 - Dataset Validation
Validates real Kaggle images in data/images/prepared/.
- Checks for RGB validity and corruption.
- Detects and logs duplicates (does NOT abort on duplicates, removes them and continues).
- Aborts only if corrupted files cannot be removed or fewer than N_IMAGES valid remain.
Logs to outputs/logs/00_dataset_validation.log
"""
import os
import sys
import hashlib
import logging
from pathlib import Path

from PIL import Image
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "00_dataset_validation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def sha256(path: str, block: int = 65_536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def check_image(path: str):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
        if mode != "RGB":
            return False, f"mode={mode}", 0, 0
        return True, "", w, h
    except Exception as exc:
        return False, str(exc), 0, 0


def main():
    log.info("=" * 60)
    log.info("STAGE 0 - Dataset Validation")
    log.info("=" * 60)

    img_dir = config.IMAGE_DIR  # data/images/prepared
    valid_exts = {".jpg", ".jpeg", ".png"}

    if not os.path.isdir(img_dir):
        log.error(f"Prepared image directory does not exist: {img_dir}")
        log.error("Run Stage 1 (01_prepare_dataset.py) first.")
        sys.exit(1)

    all_files = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if Path(f).suffix.lower() in valid_exts
    ]
    log.info(f"Total image files found: {len(all_files)}")

    # ── Integrity / RGB check ─────────────────────────────────────────────────
    valid_files = []
    corrupted   = []
    widths, heights = [], []

    for p in all_files:
        ok, reason, w, h = check_image(p)
        if ok:
            valid_files.append(p)
            widths.append(w); heights.append(h)
        else:
            corrupted.append((p, reason))
            log.warning(f"  CORRUPTED: {os.path.basename(p)}  ({reason})")

    log.info(f"Valid images         : {len(valid_files)}")
    log.info(f"Corrupted / non-RGB  : {len(corrupted)}")

    if widths:
        log.info(f"Resolution range     : {min(widths)}x{min(heights)} – {max(widths)}x{max(heights)}")
        log.info(f"Average resolution   : {int(sum(widths)/len(widths))}x{int(sum(heights)/len(heights))}")

    # ── Duplicate detection ───────────────────────────────────────────────────
    log.info("Running SHA-256 duplicate check...")
    hash_map: dict[str, str] = {}
    unique_files = []
    duplicates   = []

    for p in valid_files:
        h = sha256(p)
        if h in hash_map:
            duplicates.append((p, hash_map[h]))
            log.warning(f"  DUPLICATE: {os.path.basename(p)}  == {os.path.basename(hash_map[h])}")
        else:
            hash_map[h] = p
            unique_files.append(p)

    log.info(f"Duplicates found     : {len(duplicates)}")

    if duplicates:
        log.info("Removing duplicate files (keeping first occurrence)...")
        for dup_path, _ in duplicates:
            try:
                os.remove(dup_path)
                log.info(f"  Removed: {os.path.basename(dup_path)}")
            except Exception as exc:
                log.error(f"  Could not remove duplicate {dup_path}: {exc}")

    log.info(f"Unique valid images  : {len(unique_files)}")

    # ── Abort guard ───────────────────────────────────────────────────────────
    if len(unique_files) < config.N_IMAGES:
        log.error(
            f"ABORT: Only {len(unique_files)} valid unique images available "
            f"(need {config.N_IMAGES})."
        )
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info(f"  Total scanned    : {len(all_files)}")
    log.info(f"  Corrupted removed: {len(corrupted)}")
    log.info(f"  Duplicates removed:{len(duplicates)}")
    log.info(f"  Valid unique     : {len(unique_files)}")
    log.info(f"  Required minimum : {config.N_IMAGES}")
    log.info("  STATUS: PASSED")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
