"""
Stage 6 - Statistics Aggregation
Merges clustering metrics and similarity stats into
project_statistics.json and project_statistics.csv.
Logs to outputs/logs/06_generate_statistics.log
"""
import os
import sys
import json
import logging

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "06_generate_statistics.log")
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
    log.info("STAGE 6 - Statistics Aggregation")
    log.info("=" * 60)

    emb_path = os.path.join(config.EMBEDDING_DIR, "scene_embeddings.csv")
    if not os.path.exists(emb_path):
        log.error("scene_embeddings.csv not found.")
        sys.exit(1)

    df = pd.read_csv(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]

    stats: dict = {
        "total_images"      : int(len(df)),
        "embedding_dimension": int(len(emb_cols)),
        "number_of_clusters": int(config.N_CLUSTERS),
    }

    # Split composition
    if "split" in df.columns:
        for split, cnt in df["split"].value_counts().items():
            stats[f"split_{split}_count"] = int(cnt)

    # Similarity stats
    sim_path = os.path.join(config.SIMILARITY_DIR, "similarity_matrix.csv")
    if os.path.exists(sim_path):
        sim_df = pd.read_csv(sim_path, index_col=0)
        M = sim_df.values
        iu = np.triu_indices(M.shape[0], k=1)
        upper = M[iu]
        stats["mean_cosine_similarity"]   = round(float(np.mean(upper)),   6)
        stats["median_cosine_similarity"] = round(float(np.median(upper)), 6)
        stats["min_cosine_similarity"]    = round(float(np.min(upper)),    6)
        stats["max_cosine_similarity"]    = round(float(np.max(upper)),    6)
        log.info(f"Similarity stats computed from {sim_df.shape[0]}x{sim_df.shape[1]} matrix.")
    else:
        log.warning("similarity_matrix.csv not found - skipping similarity stats.")

    # Clustering metrics
    cm_path = os.path.join(config.STATISTICS_DIR, "clustering_metrics.csv")
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path).iloc[0].to_dict()
        stats.update(cm)
        log.info("Clustering metrics merged.")
    else:
        log.warning("clustering_metrics.csv not found - skipping.")

    # Save
    json_path = os.path.join(config.STATISTICS_DIR, "project_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)
    log.info(f"project_statistics.json saved: {json_path}")

    csv_path = os.path.join(config.STATISTICS_DIR, "project_statistics.csv")
    pd.DataFrame([stats]).to_csv(csv_path, index=False)
    log.info(f"project_statistics.csv saved: {csv_path}")

    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    log.info("=" * 60)
    log.info("STAGE 6 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
