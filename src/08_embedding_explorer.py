"""
Stage 8 - Interactive Embedding Explorer
Merges UMAP coordinates with metadata, renders an interactive
Plotly scatter saved as outputs/embedding_explorer.html.
Logs to outputs/logs/08_embedding_explorer.log
"""
import os
import sys
import logging

import pandas as pd
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "08_embedding_explorer.log")
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
    log.info("STAGE 8 - Interactive Embedding Explorer")
    log.info("=" * 60)

    red_path  = os.path.join(config.STATISTICS_DIR, "reduced_embeddings.csv")
    meta_path = os.path.join(config.DATA_DIR, "metadata.csv")

    for p in [red_path, meta_path]:
        if not os.path.exists(p):
            log.error(f"Required file not found: {p}")
            sys.exit(1)

    df_red  = pd.read_csv(red_path)
    df_meta = pd.read_csv(meta_path)

    df = df_red.merge(df_meta[["image_id", "split", "scene_group", "dominant_category",
                                "source_dataset"]], on="image_id", how="left")

    # cluster_id as string for categorical colour
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["scene_group"] = df["scene_group"].fillna(df["split"]).fillna("unknown")

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="scene_group",
        symbol="cluster_id",
        hover_name="image_id",
        hover_data={"dominant_category": True, "scene_group": True,
                    "cluster_id": True, "split": True,
                    "umap_x": ":.3f", "umap_y": ":.3f"},
        title="GeoSceneFM — Embedding Space Explorer (UMAP · Colored by Scene Group)",
        labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2",
                "scene_group": "Scene Group", "cluster_id": "Cluster"},
        template="plotly_white",
        width=1100,
        height=850,
        color_discrete_map={
            "road_scene" : "#E74C3C",
            "sky_scene"  : "#5DADE2",
            "built_env"  : "#F39C12",
            "green_space": "#27AE60",
            "mixed"      : "#8E44AD",
            "unknown"    : "#95A5A6",
        }
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.82, line=dict(width=0.8, color="DarkSlateGrey"))
    )
    fig.update_layout(
        title_font_size=16,
        legend_title_text="Scene Group / Cluster",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    out_path = os.path.join(config.OUTPUT_DIR, "embedding_explorer.html")
    fig.write_html(out_path)
    log.info(f"embedding_explorer.html saved: {out_path}")

    log.info("=" * 60)
    log.info("STAGE 8 COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
