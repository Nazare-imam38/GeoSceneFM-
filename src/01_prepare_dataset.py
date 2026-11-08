"""
Stage 1 — Dataset Preparation  (optimised for large source pools)
------------------------------------------------------------------
Strategy:
  1. Collect candidate paths from all three Kaggle source splits.
  2. Randomly shuffle candidates (seeded).
  3. Walk the shuffled list, validate each image (RGB + integrity).
     Stop as soon as N_IMAGES * OVERSAMPLE valid images are found.
  4. Deduplicate those validated candidates via SHA-256.
  5. Take the first N_IMAGES unique valid images.
  6. Copy & rename to data/images/prepared/.
  7. Write data/metadata.csv.
  8. Log everything to outputs/logs/01_prepare_dataset.log.

This avoids hashing all 10 000 source images.
"""
import os
import sys
import shutil
import random
import hashlib
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "01_prepare_dataset.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# How many candidates to validate before sampling — safety buffer
OVERSAMPLE = 3


def sha256(path: str, block: int = 65_536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def is_valid_rgb(path: str):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.mode != "RGB":
                return False, f"mode={img.mode}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def source_split(path: str) -> str:
    lower = path.replace("\\", "/").lower()
    if "/train/" in lower:
        return "train"
    if "/val/" in lower:
        return "val"
    if "/test/" in lower:
        return "test"
    return "unknown"


def main():
    log.info("=" * 60)
    log.info("STAGE 1 — Dataset Preparation")
    log.info("=" * 60)

    valid_exts = {".jpg", ".jpeg", ".png"}
    source_dirs = [config.KAGGLE_TRAIN_DIR, config.KAGGLE_VAL_DIR, config.KAGGLE_TEST_DIR]

    # 1. Collect all candidate paths
    all_paths = []
    for src in source_dirs:
        if not os.path.isdir(src):
            log.warning(f"Source directory not found, skipping: {src}")
            continue
        found = [
            os.path.join(src, f)
            for f in os.listdir(src)
            if Path(f).suffix.lower() in valid_exts
        ]
        log.info(f"  {src}: {len(found)} images")
        all_paths.extend(found)

    log.info(f"Total candidate images: {len(all_paths)}")

    if len(all_paths) < config.N_IMAGES:
        log.error(f"ABORT: Only {len(all_paths)} source images found, need {config.N_IMAGES}.")
        sys.exit(1)

    # 2. Shuffle (seeded)
    random.seed(config.RANDOM_SEED)
    random.shuffle(all_paths)

    # 3. Validate up to N_IMAGES * OVERSAMPLE candidates
    need = config.N_IMAGES * OVERSAMPLE
    log.info(f"Validating up to {need} candidates to find {config.N_IMAGES} valid unique images...")

    valid_candidates = []
    corrupted_log    = []

    for p in all_paths:
        if len(valid_candidates) >= need:
            break
        ok, reason = is_valid_rgb(p)
        if ok:
            valid_candidates.append(p)
        else:
            corrupted_log.append((os.path.basename(p), reason))
            log.warning(f"  CORRUPTED: {os.path.basename(p)}  ({reason})")

    log.info(f"Corrupted / non-RGB skipped: {len(corrupted_log)}")
    log.info(f"Valid candidates found      : {len(valid_candidates)}")

    # 4. Deduplicate validated candidates via SHA-256
    log.info("Deduplicating candidates via SHA-256...")
    hash_map: dict[str, str] = {}
    unique_candidates = []
    dup_log           = []

    for p in valid_candidates:
        h = sha256(p)
        if h in hash_map:
            dup_log.append((os.path.basename(p), os.path.basename(hash_map[h])))
            log.warning(f"  DUPLICATE: {os.path.basename(p)}  == {os.path.basename(hash_map[h])}")
        else:
            hash_map[h] = p
            unique_candidates.append(p)

    log.info(f"Duplicates removed     : {len(dup_log)}")
    log.info(f"Unique valid candidates: {len(unique_candidates)}")

    # 5. Abort guard
    if len(unique_candidates) < config.N_IMAGES:
        log.error(
            f"ABORT: Only {len(unique_candidates)} unique valid images available "
            f"(need {config.N_IMAGES}). Widen the oversample or source more images."
        )
        sys.exit(1)

    # 6. Take exactly N_IMAGES
    sampled = unique_candidates[:config.N_IMAGES]
    log.info(f"Sampled {len(sampled)} images.")

    # 7. Clear prepared dir
    if os.path.isdir(config.IMAGE_DIR):
        old = [f for f in os.listdir(config.IMAGE_DIR) if Path(f).suffix.lower() in valid_exts]
        for f in old:
            os.remove(os.path.join(config.IMAGE_DIR, f))
        if old:
            log.info(f"Cleared {len(old)} old files from {config.IMAGE_DIR}")

    # 8. Copy & rename
    metadata = []
    split_counts: dict[str, int] = {}

    for idx, src_path in enumerate(sampled, start=1):
        new_name = f"img_{idx:03d}.jpg"
        dst_path = os.path.join(config.IMAGE_DIR, new_name)
        shutil.copy2(src_path, dst_path)

        sp = source_split(src_path)
        split_counts[sp] = split_counts.get(sp, 0) + 1

        metadata.append({
            "image_id"      : f"img_{idx:03d}",
            "filename"      : new_name,
            "filepath"      : dst_path,
            "source_path"   : src_path,
            "split"         : sp,
            "source_dataset": "Kaggle (Mapillary Vistas subset)",
            "scene_label"   : sp,
        })

    # 9. Save metadata
    df = pd.DataFrame(metadata)
    meta_path = os.path.join(config.DATA_DIR, "metadata.csv")
    df.to_csv(meta_path, index=False)
    log.info(f"metadata.csv written: {meta_path}")

    log.info("Dataset composition by source split:")
    for k, v in split_counts.items():
        log.info(f"  {k}: {v} images")

    log.info("=" * 60)
    log.info(f"STAGE 1 COMPLETE — {len(sampled)} images in {config.IMAGE_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
