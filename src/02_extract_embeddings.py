"""
Stage 2 - CLIP Embedding Extraction
Reads data/metadata.csv, runs each image through CLIP ViT-B/32,
L2-normalizes the 512-dim output, saves scene_embeddings/scene_embeddings.csv.
Logs to outputs/logs/02_extract_embeddings.log
"""
import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "02_extract_embeddings.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("STAGE 2 - CLIP Embedding Extraction")
    log.info("=" * 60)

    meta_path = os.path.join(config.DATA_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        log.error(f"metadata.csv not found at {meta_path}. Run Stage 1 first.")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    log.info(f"Loaded {len(df)} rows from metadata.csv")

    # Validate every filepath exists
    missing = df[~df["filepath"].apply(os.path.exists)]
    if len(missing) > 0:
        log.error(f"{len(missing)} filepaths in metadata.csv do not exist on disk:")
        for p in missing["filepath"].tolist():
            log.error(f"  MISSING: {p}")
        sys.exit(1)
    log.info("All filepaths validated - every image file exists on disk.")

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Loading model: {config.MODEL_NAME}")
    model = CLIPModel.from_pretrained(config.MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
    model.eval()
    log.info("Model loaded.")

    embeddings = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        image_path = row["filepath"]
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                features = model.get_image_features(**inputs)

            # Handle both tensor and dataclass returns across transformers versions
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "pooler_output"):
                    features = features.pooler_output
                else:
                    features = features[0]

            # L2 normalize onto unit sphere
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            vec = features.cpu().numpy()[0]

            row_dict = {
                "image_id" : row["image_id"],
                "filename" : row["filename"],
                "filepath" : row["filepath"],
                "split"    : row.get("split", "unknown"),
            }
            for i, v in enumerate(vec):
                row_dict[f"embedding_{i}"] = float(v)

            embeddings.append(row_dict)

        except Exception as exc:
            log.warning(f"  FAILED {os.path.basename(image_path)}: {exc}")
            failed.append({"image_id": row["image_id"], "filepath": image_path, "error": str(exc)})

    log.info(f"Embeddings extracted: {len(embeddings)}")
    log.info(f"Failed:               {len(failed)}")

    if len(embeddings) == 0:
        log.error("No embeddings produced. Aborting.")
        sys.exit(1)

    emb_df = pd.DataFrame(embeddings)
    emb_path = os.path.join(config.EMBEDDING_DIR, "scene_embeddings.csv")
    emb_df.to_csv(emb_path, index=False)
    log.info(f"scene_embeddings.csv saved: {emb_path}  ({len(emb_df)} rows)")

    # Verify no synthetic image references remain
    synth_refs = emb_df[emb_df["filepath"].str.contains(r"(residential|commercial|downtown|park)_\d+", regex=True, na=False)]
    if len(synth_refs) > 0:
        log.error(f"SYNTHETIC IMAGE REFERENCES found in embeddings ({len(synth_refs)} rows). Aborting.")
        sys.exit(1)
    log.info("Synthetic image check: PASSED - no synthetic references in embeddings.")

    vec_dim = len([c for c in emb_df.columns if c.startswith("embedding_")])

    meta_info = {
        "model_name"              : config.MODEL_NAME,
        "scene_embedding_dimension": vec_dim,
        "number_of_images"        : len(emb_df),
        "failed_images"           : len(failed),
        "date_created"            : datetime.now().isoformat(),
        "random_seed"             : config.RANDOM_SEED,
        "device"                  : device,
    }
    meta_out = os.path.join(config.STATISTICS_DIR, "embedding_metadata.json")
    with open(meta_out, "w") as f:
        json.dump(meta_info, f, indent=4)
    log.info(f"embedding_metadata.json saved: {meta_out}")

    if failed:
        fail_path = os.path.join(config.LOG_DIR, "02_failed_images.log")
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        log.warning(f"Failed image list saved: {fail_path}")

    log.info("=" * 60)
    log.info(f"STAGE 2 COMPLETE - {len(emb_df)} embeddings, dim={vec_dim}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
