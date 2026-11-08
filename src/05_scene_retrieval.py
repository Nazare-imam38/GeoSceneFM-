"""
Stage 5 - Scene Retrieval
Computes full pairwise cosine similarity matrix,
finds top-5 nearest neighbours per image,
generates heatmap and a retrieval gallery.
Logs to outputs/logs/05_scene_retrieval.log
"""
import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "05_scene_retrieval.log")
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
    log.info("STAGE 5 - Scene Retrieval")
    log.info("=" * 60)

    emb_path = os.path.join(config.EMBEDDING_DIR, "scene_embeddings.csv")
    if not os.path.exists(emb_path):
        log.error(f"scene_embeddings.csv not found: {emb_path}")
        sys.exit(1)

    df = pd.read_csv(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    X = df[emb_cols].values
    log.info(f"Computing {len(df)}x{len(df)} cosine similarity matrix...")

    sim = cosine_similarity(X)
    log.info(f"Similarity matrix computed. Shape: {sim.shape}")

    # ── Save full matrix ──────────────────────────────────────────────────────
    sim_df = pd.DataFrame(sim, index=df["image_id"], columns=df["image_id"])
    sim_csv = os.path.join(config.SIMILARITY_DIR, "similarity_matrix.csv")
    sim_df.to_csv(sim_csv)
    log.info(f"similarity_matrix.csv saved: {sim_csv}")

    # ── Top-5 nearest neighbours ──────────────────────────────────────────────
    log.info("Computing top-5 nearest neighbours per image...")
    records = []
    for i in range(len(df)):
        row_sim = sim[i].copy()
        row_sim[i] = -1.0          # exclude self
        top5 = np.argsort(row_sim)[::-1][:5]
        for rank, j in enumerate(top5, start=1):
            records.append({
                "query_image_id"   : df.iloc[i]["image_id"],
                "similar_image_id" : df.iloc[j]["image_id"],
                "rank"             : rank,
                "cosine_similarity": float(sim[i, j]),
                "query_filepath"   : df.iloc[i]["filepath"],
                "similar_filepath" : df.iloc[j]["filepath"],
            })

    top5_df = pd.DataFrame(records)
    top5_path = os.path.join(config.SIMILARITY_DIR, "top5_similar_scenes.csv")
    top5_df.to_csv(top5_path, index=False)
    log.info(f"top5_similar_scenes.csv saved: {top5_path}  ({len(top5_df)} rows)")

    # ── Similarity statistics ─────────────────────────────────────────────────
    iu = np.triu_indices(len(df), k=1)
    upper = sim[iu]
    log.info(f"Pairwise similarity  mean={upper.mean():.4f}  "
             f"median={np.median(upper):.4f}  "
             f"min={upper.min():.4f}  max={upper.max():.4f}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Cosine Similarity Heatmap (All Scene Pairs)", fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(config.FIGURE_DIR, "similarity_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    log.info(f"Saved: {heatmap_path}")

    # ── Retrieval gallery (5 random queries) ─────────────────────────────────
    gallery_dir = os.path.join(config.OUTPUT_DIR, "retrieval_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    np.random.seed(config.RANDOM_SEED)
    query_ids = np.random.choice(df["image_id"].values, size=min(5, len(df)), replace=False)

    for q_num, q_id in enumerate(query_ids, start=1):
        q_rows = top5_df[top5_df["query_image_id"] == q_id].sort_values("rank").head(5)
        q_path = q_rows.iloc[0]["query_filepath"]

        fig, axes = plt.subplots(1, 6, figsize=(24, 4))

        # Query image
        try:
            axes[0].imshow(Image.open(q_path))
        except Exception:
            axes[0].text(0.5, 0.5, "N/A", ha="center")
        axes[0].set_title(f"Query\n{q_id}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # Top-5
        for r_idx, (_, row) in enumerate(q_rows.iterrows()):
            ax = axes[r_idx + 1]
            try:
                ax.imshow(Image.open(row["similar_filepath"]))
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center")
            ax.set_title(f"Rank {row['rank']}\n{row['similar_image_id']}\nsim={row['cosine_similarity']:.3f}",
                         fontsize=8)
            ax.axis("off")

        plt.suptitle(f"Retrieval Gallery — Query {q_num}", fontsize=13)
        plt.tight_layout()
        out = os.path.join(gallery_dir, f"query_{q_num:02d}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"Saved gallery: {out}")

    log.info("=" * 60)
    log.info("STAGE 5 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
