"""
Stage 9 - Advanced Embedding Analysis
- Cluster summary (dominant split, count, mean intra-cluster similarity)
- Cluster representative image panels (top-5 most central per cluster)
- Embedding statistics (norm, variance, entropy, pairwise similarity)
- Similarity network graph (NetworkX, threshold 0.85)
Logs to outputs/logs/09_embedding_analysis.log
"""
import os
import sys
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "09_embedding_analysis.log")
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
    log.info("STAGE 9 - Advanced Embedding Analysis")
    log.info("=" * 60)

    emb_path  = os.path.join(config.EMBEDDING_DIR, "scene_embeddings.csv")
    meta_path = os.path.join(config.DATA_DIR, "metadata.csv")
    red_path  = os.path.join(config.STATISTICS_DIR, "reduced_embeddings.csv")

    for p in [emb_path, meta_path, red_path]:
        if not os.path.exists(p):
            log.error(f"Required file missing: {p}")
            sys.exit(1)

    df_emb  = pd.read_csv(emb_path)
    df_meta = pd.read_csv(meta_path)
    df_red  = pd.read_csv(red_path)

    df = df_emb.copy()
    if "scene_group" not in df.columns and "scene_group" in df_meta.columns:
        df = df.merge(df_meta[["image_id", "split", "scene_group", "dominant_category"]], on="image_id", how="left")
    elif "split" not in df.columns and "split" in df_meta.columns:
        df = df.merge(df_meta[["image_id", "split"]], on="image_id", how="left")
    if "cluster_id" not in df.columns:
        df = df.merge(df_red[["image_id", "cluster_id"]], on="image_id", how="left")
    if "split" not in df.columns:
        df["split"] = "unknown"
    if "scene_group" not in df.columns:
        df["scene_group"] = df.get("split", "unknown")
    # Use scene_group as the primary label
    df["label"] = df["scene_group"].fillna(df["split"]).fillna("unknown")

    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    X = df[emb_cols].values

    log.info(f"Building {len(df)}x{len(df)} cosine similarity matrix...")
    sim = cosine_similarity(X)

    # ── 1. Cluster Summary ────────────────────────────────────────────────────
    log.info("Computing cluster summary...")
    clusters = sorted(df["cluster_id"].dropna().unique())
    summary = []
    for c in clusters:
        mask = df["cluster_id"] == c
        c_df = df[mask]
        c_idx = c_df.index.tolist()

        dominant = c_df["label"].value_counts().idxmax() if len(c_df) > 0 else "N/A"

        if len(c_idx) > 1:
            sub = sim[np.ix_(c_idx, c_idx)]
            iu = np.triu_indices(len(c_idx), k=1)
            mean_sim = float(np.mean(sub[iu]))
        else:
            mean_sim = 1.0

        summary.append({
            "Cluster"         : int(c),
            "Dominant Split"  : dominant,
            "Images"          : int(len(c_df)),
            "Mean Intra-Sim"  : round(mean_sim, 4),
        })
        log.info(f"  Cluster {int(c)}: {len(c_df)} images, dominant={dominant}, mean_sim={mean_sim:.4f}")

    pd.DataFrame(summary).to_csv(
        os.path.join(config.STATISTICS_DIR, "cluster_summary.csv"), index=False
    )
    log.info("cluster_summary.csv saved.")

    # ── 2. Representative Image Panels ────────────────────────────────────────
    log.info("Generating cluster representative image panels...")
    examples_dir = os.path.join(config.OUTPUT_DIR, "cluster_examples")
    os.makedirs(examples_dir, exist_ok=True)

    for c in clusters:
        mask = df["cluster_id"] == c
        c_df = df[mask]
        c_idx = c_df.index.tolist()
        if len(c_idx) == 0:
            continue

        # Rank by centrality (sum of intra-cluster similarities)
        sub = sim[np.ix_(c_idx, c_idx)]
        centrality = sub.sum(axis=1)
        top_local = np.argsort(centrality)[::-1][:min(5, len(c_idx))]
        top_global = [c_idx[i] for i in top_local]

        n = len(top_global)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Cluster {int(c)} — Most Central Images", fontsize=14)

        for ax, idx in zip(axes, top_global):
            img_path = df.iloc[idx]["filepath"]
            try:
                ax.imshow(Image.open(img_path))
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(df.iloc[idx]["image_id"], fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        out = os.path.join(examples_dir, f"cluster_{int(c)}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        log.info(f"  Saved: {out}")

    # ── 3. Embedding Statistics ───────────────────────────────────────────────
    log.info("Computing embedding statistics...")
    norms     = np.linalg.norm(X, axis=1)
    variances = np.var(X, axis=0)
    iu        = np.triu_indices(len(df), k=1)
    all_sims  = sim[iu]

    split_counts = df["label"].value_counts()
    se = float(entropy(split_counts.values))

    emb_stats = {
        "mean_embedding_norm"    : round(float(np.mean(norms)), 6),
        "std_embedding_norm"     : round(float(np.std(norms)), 6),
        "mean_feature_variance"  : round(float(np.mean(variances)), 8),
        "mean_cosine_similarity" : round(float(np.mean(all_sims)), 6),
        "scene_group_entropy"    : round(se, 6),
        "total_images"           : int(len(df)),
    }
    pd.DataFrame([emb_stats]).to_csv(
        os.path.join(config.STATISTICS_DIR, "embedding_statistics.csv"), index=False
    )
    for k, v in emb_stats.items():
        log.info(f"  {k}: {v}")
    log.info("embedding_statistics.csv saved.")

    # ── 4. Similarity Network Graph ───────────────────────────────────────────
    log.info("Building similarity network graph (threshold=0.85)...")
    G = nx.Graph()

    for i in range(len(df)):
        G.add_node(df.iloc[i]["image_id"], cluster=int(df.iloc[i]["cluster_id"]))

    edge_count = 0
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim[i, j] > 0.85:
                G.add_edge(
                    df.iloc[i]["image_id"],
                    df.iloc[j]["image_id"],
                    weight=float(sim[i, j]),
                )
                edge_count += 1

    log.info(f"Network: {G.number_of_nodes()} nodes, {edge_count} edges (sim > 0.85)")

    palette = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD",
               "#F39C12", "#16A085", "#D35400", "#7F8C8D"]
    node_colors = [palette[int(G.nodes[n]["cluster"]) % len(palette)] for n in G.nodes]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=config.RANDOM_SEED, k=0.18)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.18, edge_color="gray", width=0.7)

    handles = [plt.scatter([], [], color=palette[int(c) % len(palette)],
                           label=f"Cluster {int(c)}", s=60)
               for c in clusters]
    plt.legend(handles=handles, title="Cluster", loc="best", fontsize=9)
    plt.title("Embedding Similarity Network (Cosine Similarity > 0.85)", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    net_path = os.path.join(config.FIGURE_DIR, "embedding_similarity_network.png")
    plt.savefig(net_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {net_path}")

    log.info("=" * 60)
    log.info("STAGE 9 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
