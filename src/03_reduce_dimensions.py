"""
Stage 3 - Dimensionality Reduction
Loads scene_embeddings.csv, runs PCA / UMAP / t-SNE,
saves reduced_embeddings.csv and three scatter-plot PNGs.
Logs to outputs/logs/03_reduce_dimensions.log
"""
import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "03_reduce_dimensions.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def scatter_plot(x, y, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, alpha=0.75, s=55, c="steelblue", edgecolors="white", linewidths=0.4)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info(f"Saved: {out_path}")


def main():
    log.info("=" * 60)
    log.info("STAGE 3 - Dimensionality Reduction")
    log.info("=" * 60)

    emb_path = os.path.join(config.EMBEDDING_DIR, "scene_embeddings.csv")
    if not os.path.exists(emb_path):
        log.error(f"scene_embeddings.csv not found: {emb_path}. Run Stage 2 first.")
        sys.exit(1)

    df = pd.read_csv(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    X = df[emb_cols].values
    log.info(f"Loaded {X.shape[0]} embeddings of dimension {X.shape[1]}")

    # ── PCA ─────────────────────────────────────────────────────────────────
    log.info("Running PCA (n_components=2)...")
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    pca_res = pca.fit_transform(X)
    df["pca_x"] = pca_res[:, 0]
    df["pca_y"] = pca_res[:, 1]
    ev = pca.explained_variance_ratio_
    log.info(f"PCA explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}, total={sum(ev):.3f}")

    # ── UMAP ────────────────────────────────────────────────────────────────
    log.info("Running UMAP (n_components=2)...")
    reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED)
    umap_res = reducer.fit_transform(X)
    df["umap_x"] = umap_res[:, 0]
    df["umap_y"] = umap_res[:, 1]
    log.info("UMAP complete.")

    # ── t-SNE ───────────────────────────────────────────────────────────────
    perplexity = min(30, max(5, X.shape[0] // 3))
    log.info(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=config.RANDOM_SEED)
    tsne_res = tsne.fit_transform(X)
    df["tsne_x"] = tsne_res[:, 0]
    df["tsne_y"] = tsne_res[:, 1]
    log.info("t-SNE complete.")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_cols = ["image_id", "pca_x", "pca_y", "umap_x", "umap_y", "tsne_x", "tsne_y"]
    out_path = os.path.join(config.STATISTICS_DIR, "reduced_embeddings.csv")
    df[out_cols].to_csv(out_path, index=False)
    log.info(f"reduced_embeddings.csv saved: {out_path}")

    # ── Validate ─────────────────────────────────────────────────────────────
    assert len(df) == X.shape[0], "Row count mismatch after reduction"
    assert not df[["pca_x", "pca_y", "umap_x", "umap_y", "tsne_x", "tsne_y"]].isnull().any().any(), \
        "NaN values found in reduced embeddings"
    log.info("Validation PASSED: reduced_embeddings.csv is complete and NaN-free.")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    scatter_plot(df["pca_x"],  df["pca_y"],
                 "PCA Projection of Scene Embeddings",
                 "PC 1", "PC 2",
                 os.path.join(config.FIGURE_DIR, "pca_projection.png"))

    scatter_plot(df["umap_x"], df["umap_y"],
                 "UMAP Projection of Scene Embeddings",
                 "UMAP 1", "UMAP 2",
                 os.path.join(config.FIGURE_DIR, "umap_projection.png"))

    scatter_plot(df["tsne_x"], df["tsne_y"],
                 "t-SNE Projection of Scene Embeddings",
                 "t-SNE 1", "t-SNE 2",
                 os.path.join(config.FIGURE_DIR, "tsne_projection.png"))

    log.info("=" * 60)
    log.info("STAGE 3 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
