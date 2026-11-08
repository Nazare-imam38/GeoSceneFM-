"""
Stage 7 - Static Visualizations
Creates pipeline diagram, embedding quality dashboard,
and cosine similarity distribution histogram.
Logs to outputs/logs/07_create_visualizations.log
"""
import os
import sys
import json
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "07_create_visualizations.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def pipeline_diagram():
    steps = [
        "Real Kaggle Urban Images",
        "CLIP ViT-B/32 (Zero-Shot Encoder)",
        "512-Dim L2-Normalized Embeddings",
        "Dimensionality Reduction (PCA / UMAP / t-SNE)",
        "Cosine Similarity Computation",
        "Semantic Scene Retrieval (Top-5 NN)",
        "KMeans Clustering (K=4 Typologies)",
        "Spatial Foundation Model Prototype",
    ]
    colors = [
        "#AED6F1", "#85C1E9", "#5DADE2", "#2E86C1",
        "#1A5276", "#154360", "#0D2137", "#082032",
    ]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y_positions = np.linspace(0.92, 0.08, len(steps))

    for i, (step, color) in enumerate(zip(steps, colors)):
        rect = mpatches.FancyBboxPatch(
            (0.15, y_positions[i] - 0.038), 0.7, 0.076,
            boxstyle="round,pad=0.01",
            linewidth=1.5, edgecolor="#1A252F", facecolor=color,
        )
        ax.add_patch(rect)
        text_color = "white" if i >= 4 else "black"
        ax.text(0.5, y_positions[i], step,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=text_color)
        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(0.5, y_positions[i + 1] + 0.038),
                xytext=(0.5, y_positions[i] - 0.038),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.8),
            )

    plt.title("GeoSceneFM Representation Learning Pipeline", fontsize=15, pad=12)
    out = os.path.join(config.FIGURE_DIR, "embedding_pipeline_diagram.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out}")


def quality_dashboard():
    stats_path = os.path.join(config.STATISTICS_DIR, "project_statistics.json")
    if not os.path.exists(stats_path):
        log.warning("project_statistics.json missing - skipping dashboard.")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    rows = [
        ("Total Images",             stats.get("total_images",             "N/A")),
        ("Embedding Dimension",       stats.get("embedding_dimension",      "N/A")),
        ("Number of Clusters",        stats.get("number_of_clusters",       "N/A")),
        ("Mean Cosine Similarity",    f"{stats.get('mean_cosine_similarity', 0):.4f}"),
        ("Median Cosine Similarity",  f"{stats.get('median_cosine_similarity', 0):.4f}"),
        ("Silhouette Score",          f"{stats.get('silhouette_score', 0):.4f}"),
        ("Davies-Bouldin Index",      f"{stats.get('davies_bouldin_index', 0):.4f}"),
        ("Calinski-Harabasz Score",   f"{stats.get('calinski_harabasz_score', 0):.1f}"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.5, 0.97, "GeoSceneFM — Embedding Quality Dashboard",
            fontsize=17, fontweight="bold", ha="center", va="top",
            transform=ax.transAxes)

    y = 0.83
    for label, val in rows:
        ax.text(0.12, y, f"{label}:", fontsize=13, fontweight="bold",
                ha="left", va="center", transform=ax.transAxes)
        ax.text(0.62, y, str(val), fontsize=13,
                ha="left", va="center", transform=ax.transAxes)
        ax.plot([0.08, 0.92], [y - 0.04, y - 0.04],
                color="#BDC3C7", linewidth=0.6,
                transform=ax.transAxes)
        y -= 0.10

    out = os.path.join(config.FIGURE_DIR, "embedding_quality_dashboard.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {out}")


def similarity_distribution():
    sim_path = os.path.join(config.SIMILARITY_DIR, "similarity_matrix.csv")
    if not os.path.exists(sim_path):
        log.warning("similarity_matrix.csv missing - skipping distribution plot.")
        return

    sim_df = pd.read_csv(sim_path, index_col=0)
    M = sim_df.values
    iu = np.triu_indices(M.shape[0], k=1)
    upper = M[iu]

    plt.figure(figsize=(9, 6))
    sns.histplot(upper, bins=35, kde=True, color="#5DADE2", edgecolor="white")
    plt.axvline(upper.mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean = {upper.mean():.3f}")
    plt.title("Pairwise Cosine Similarity Distribution", fontsize=16)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(config.FIGURE_DIR, "cosine_similarity_distribution.png")
    plt.savefig(out, dpi=300)
    plt.close()
    log.info(f"Saved: {out}")


def main():
    log.info("=" * 60)
    log.info("STAGE 7 - Static Visualizations")
    log.info("=" * 60)

    pipeline_diagram()
    quality_dashboard()
    similarity_distribution()

    log.info("=" * 60)
    log.info("STAGE 7 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
