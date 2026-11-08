"""
Stage 4 - KMeans Clustering
Clusters the 512-dim embedding space into N_CLUSTERS groups,
evaluates with Silhouette / Davies-Bouldin / Calinski-Harabasz,
and produces PCA/UMAP cluster scatter plots and a bar chart.
Logs to outputs/logs/04_cluster_embeddings.log
"""
import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "04_cluster_embeddings.log")
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
    log.info("STAGE 4 - KMeans Clustering")
    log.info("=" * 60)

    emb_path = os.path.join(config.EMBEDDING_DIR, "scene_embeddings.csv")
    red_path = os.path.join(config.STATISTICS_DIR, "reduced_embeddings.csv")

    for p in [emb_path, red_path]:
        if not os.path.exists(p):
            log.error(f"Required file not found: {p}")
            sys.exit(1)

    df_emb = pd.read_csv(emb_path)
    df_red = pd.read_csv(red_path)

    emb_cols = [c for c in df_emb.columns if c.startswith("embedding_")]
    X = df_emb[emb_cols].values
    log.info(f"Embedding matrix: {X.shape}")

    # ── KMeans ──────────────────────────────────────────────────────────────
    log.info(f"Running KMeans (n_clusters={config.N_CLUSTERS}, n_init=auto)...")
    km = KMeans(
        n_clusters=config.N_CLUSTERS,
        random_state=config.RANDOM_SEED,
        n_init="auto",
    )
    labels = km.fit_predict(X)

    df_emb["cluster_id"] = labels
    df_red["cluster_id"] = labels

    # ── Metrics ──────────────────────────────────────────────────────────────
    n_unique = len(set(labels))
    if n_unique > 1:
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
    else:
        sil = db = ch = float("nan")

    log.info(f"Silhouette Score     : {sil:.4f}  (higher = better, max 1.0)")
    log.info(f"Davies-Bouldin Index : {db:.4f}  (lower  = better, min 0.0)")
    log.info(f"Calinski-Harabasz    : {ch:.4f}  (higher = better)")

    cluster_counts = df_emb["cluster_id"].value_counts().sort_index().to_dict()
    log.info("Cluster sizes:")
    for k, v in cluster_counts.items():
        log.info(f"  Cluster {k}: {v} images")

    # ── Save outputs ─────────────────────────────────────────────────────────
    assign_path = os.path.join(config.STATISTICS_DIR, "cluster_assignments.csv")
    df_emb[["image_id", "cluster_id"]].to_csv(assign_path, index=False)
    log.info(f"cluster_assignments.csv saved: {assign_path}")

    df_red.to_csv(red_path, index=False)
    log.info(f"reduced_embeddings.csv updated with cluster_id: {red_path}")

    metrics = {
        "silhouette_score"      : round(sil, 6),
        "davies_bouldin_index"  : round(db, 6),
        "calinski_harabasz_score": round(ch, 6),
    }
    metrics.update({f"cluster_{k}_size": v for k, v in cluster_counts.items()})
    pd.DataFrame([metrics]).to_csv(
        os.path.join(config.STATISTICS_DIR, "clustering_metrics.csv"), index=False
    )
    log.info("clustering_metrics.csv saved.")

    # ── Figures ───────────────────────────────────────────────────────────────
    palette = "tab10"

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_red, x="pca_x", y="pca_y",
        hue="cluster_id", palette=palette, s=70, alpha=0.85,
    )
    plt.title("PCA Projection — Urban Scene Clusters", fontsize=16)
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    fig_pca = os.path.join(config.FIGURE_DIR, "pca_clusters.png")
    plt.savefig(fig_pca, dpi=300)
    plt.close()
    log.info(f"Saved: {fig_pca}")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_red, x="umap_x", y="umap_y",
        hue="cluster_id", palette=palette, s=70, alpha=0.85,
    )
    plt.title("UMAP Projection — Urban Scene Typologies", fontsize=16)
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    fig_umap = os.path.join(config.FIGURE_DIR, "umap_clusters.png")
    plt.savefig(fig_umap, dpi=300)
    plt.close()
    log.info(f"Saved: {fig_umap}")

    plt.figure(figsize=(8, 6))
    order = sorted(df_emb["cluster_id"].unique())
    sns.countplot(data=df_emb, x="cluster_id", hue="cluster_id",
                  palette=palette, order=order, legend=False)
    plt.title("Images per Urban Scene Typology (Cluster)", fontsize=16)
    plt.xlabel("Cluster ID"); plt.ylabel("Count")
    plt.tight_layout()
    fig_bar = os.path.join(config.FIGURE_DIR, "cluster_size_bar.png")
    plt.savefig(fig_bar, dpi=300)
    plt.close()
    log.info(f"Saved: {fig_bar}")

    log.info("=" * 60)
    log.info(f"STAGE 4 COMPLETE  Sil={sil:.3f}  DB={db:.3f}  CH={ch:.1f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
