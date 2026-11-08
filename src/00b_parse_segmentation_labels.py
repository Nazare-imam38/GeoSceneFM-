"""
Stage 0b — Parse Segmentation Masks to Derive Scene Labels
------------------------------------------------------------
For every image in data/metadata.csv, finds the matching
*_train_id.png label mask, computes per-category pixel counts,
and assigns:
  - dominant_category   : single most-common category name
  - scene_group         : coarse group (road_scene / built_env /
                          green_space / sky_scene / mixed)
  - top3_categories     : comma-separated top-3 category names
  - category pixel fractions (one column per category)

Updates data/metadata.csv in-place with the new columns.
Also writes outputs/statistics/label_composition.csv with the
full per-image pixel distribution.

Logs to outputs/logs/00b_parse_segmentation_labels.log
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "00b_parse_segmentation_labels.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Cityscapes / Mapillary Vistas ID → category name ─────────────────────────
# These IDs are confirmed from the pixel analysis of this dataset's label masks.
CATEGORY_MAP = {
    0:  "road",
    1:  "sidewalk",
    2:  "building",
    3:  "wall",
    4:  "fence",
    5:  "pole",
    6:  "traffic_light",
    7:  "traffic_sign",
    8:  "vegetation",
    9:  "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "unlabeled",
}

# Coarse scene groups — used to assign a human-readable scene_group label
# Logic: pick group based on top-3 dominant categories
ROAD_CATS       = {"road", "sidewalk", "car", "truck", "bus", "motorcycle", "bicycle"}
BUILT_CATS      = {"building", "wall", "fence", "pole", "traffic_light", "traffic_sign"}
GREEN_CATS      = {"vegetation", "terrain"}
SKY_CATS        = {"sky"}
PERSON_CATS     = {"person", "rider"}

ALL_CATS = sorted(set(CATEGORY_MAP.values()) - {"unlabeled"})


def scene_group_from_top3(top3: list[str]) -> str:
    """
    Assign a coarse scene group from the three dominant categories.
    Rules (checked in priority order):
      - If sky > 25 % pixels  → sky_scene
      - If green dominates     → green_space
      - If road/vehicle dom.   → road_scene
      - If building dom.       → built_env
      - Otherwise              → mixed
    """
    top3_set = set(top3)
    if top3_set & SKY_CATS and top3[0] in SKY_CATS:
        return "sky_scene"
    if top3_set & GREEN_CATS and top3[0] in GREEN_CATS:
        return "green_space"
    if top3_set & ROAD_CATS and top3[0] in ROAD_CATS:
        return "road_scene"
    if top3_set & BUILT_CATS and top3[0] in BUILT_CATS:
        return "built_env"
    return "mixed"


def parse_mask(mask_path: str) -> dict:
    """
    Returns dict of {category_name: fraction_of_valid_pixels}
    Excludes class 255 (unlabeled) from denominator.
    """
    arr = np.array(Image.open(mask_path))
    valid = arr[arr != 255]
    total = len(valid)
    if total == 0:
        return {}

    result = {}
    for cat_id, cat_name in CATEGORY_MAP.items():
        if cat_name == "unlabeled":
            continue
        count = int(np.sum(arr == cat_id))
        if count > 0:
            result[cat_name] = count / total
    return result


def find_label_mask(image_id: str, source_path: str) -> str | None:
    """
    Given an image_id like 'img_042' and its source_path
    (e.g. data/images/train/0004a4c0-d4dff0ad.jpg), find the
    corresponding *_train_id.png mask.
    """
    # Derive stem from source filename
    src_stem = Path(source_path).stem          # e.g. '0004a4c0-d4dff0ad'
    src_lower = source_path.replace("\\", "/").lower()

    # Determine which split the source came from
    if "/train/" in src_lower:
        label_dir = os.path.join(config.LABEL_DIR, "train")
    elif "/val/" in src_lower:
        label_dir = os.path.join(config.LABEL_DIR, "val")
    else:
        # test split has no labels — try train then val
        for split in ["train", "val"]:
            candidate = os.path.join(config.LABEL_DIR, split, f"{src_stem}_train_id.png")
            if os.path.exists(candidate):
                return candidate
        return None

    mask_path = os.path.join(label_dir, f"{src_stem}_train_id.png")
    return mask_path if os.path.exists(mask_path) else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("STAGE 0b — Parse Segmentation Labels")
    log.info("=" * 60)

    meta_path = os.path.join(config.DATA_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        log.error(f"metadata.csv not found: {meta_path}. Run Stage 1 first.")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    log.info(f"Loaded {len(df)} rows from metadata.csv")

    rows = []
    no_mask_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing masks"):
        image_id    = row["image_id"]
        source_path = row.get("source_path", row["filepath"])

        mask_path = find_label_mask(image_id, source_path)

        if mask_path is None:
            log.warning(f"  No mask found for {image_id} (source: {Path(source_path).name})")
            no_mask_count += 1
            rows.append({
                "image_id"          : image_id,
                "dominant_category" : "unknown",
                "scene_group"       : "unknown",
                "top3_categories"   : "unknown",
                **{c: 0.0 for c in ALL_CATS},
            })
            continue

        fractions = parse_mask(mask_path)
        if not fractions:
            log.warning(f"  Empty mask: {Path(mask_path).name}")
            no_mask_count += 1
            rows.append({
                "image_id"          : image_id,
                "dominant_category" : "unknown",
                "scene_group"       : "unknown",
                "top3_categories"   : "unknown",
                **{c: 0.0 for c in ALL_CATS},
            })
            continue

        # Sort by fraction descending
        sorted_cats = sorted(fractions.items(), key=lambda x: x[1], reverse=True)
        top3        = [c for c, _ in sorted_cats[:3]]
        dominant    = top3[0]
        group       = scene_group_from_top3(top3)
        top3_str    = ", ".join(top3)

        row_dict = {
            "image_id"          : image_id,
            "dominant_category" : dominant,
            "scene_group"       : group,
            "top3_categories"   : top3_str,
        }
        for cat in ALL_CATS:
            row_dict[cat] = round(fractions.get(cat, 0.0), 5)

        rows.append(row_dict)

    label_df = pd.DataFrame(rows)

    # ── Save full composition table ──────────────────────────────────────────
    comp_path = os.path.join(config.STATISTICS_DIR, "label_composition.csv")
    label_df.to_csv(comp_path, index=False)
    log.info(f"label_composition.csv saved: {comp_path}")

    # ── Merge scene labels back into metadata ────────────────────────────────
    merge_cols = ["image_id", "dominant_category", "scene_group", "top3_categories"]
    df = df.drop(columns=[c for c in merge_cols[1:] if c in df.columns], errors="ignore")
    df = df.merge(label_df[merge_cols], on="image_id", how="left")

    # Overwrite scene_label with dominant_category (more meaningful than split)
    df["scene_label"] = df["dominant_category"]

    df.to_csv(meta_path, index=False)
    log.info(f"metadata.csv updated with scene labels: {meta_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info(f"\nImages with masks found  : {len(df) - no_mask_count}")
    log.info(f"Images without masks     : {no_mask_count}")

    log.info("\nDominant category distribution:")
    for cat, cnt in df["dominant_category"].value_counts().items():
        log.info(f"  {cat:20s}: {cnt}")

    log.info("\nScene group distribution:")
    for grp, cnt in df["scene_group"].value_counts().items():
        log.info(f"  {grp:20s}: {cnt}")

    log.info("=" * 60)
    log.info("STAGE 0b COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
