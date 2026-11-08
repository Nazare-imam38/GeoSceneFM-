"""
Stage 10 — Cluster Purity Evaluation
--------------------------------------
Computes formal cluster purity against segmentation-derived scene_group labels.
Also computes NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index).
Produces:
  - outputs/statistics/cluster_purity.csv      — per-cluster breakdown
  - outputs/statistics/purity_summary.json     — headline numbers
  - outputs/figures/cluster_purity_heatmap.png — cluster × scene_group confusion matrix
  - outputs/figures/cluster_scene_composition.png — stacked bar per cluster
Logs to outputs/logs/10_cluster_purity.log
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_PATH = os.path.join(config.LOG_DIR, "10_cluster_purity.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SCENE_COLORS = {
    "road_scene" : "#E74C3C",
    "sky_scene"  : "#5DADE2",
    "built_env"  : "#F39C12",
    "green_space": "#27AE60",
    "unknown"    : "#95A5A6",
    "mixed"      : "#8E44AD",
}


def main():
    log.info("=" * 60)
    log.info("STAGE 10 — Cluster Purity Evaluation")
    log.info("=" * 60)

    meta_path  = os.path.join(config.DATA_DIR, "metadata.csv")
    assign_path = os.path.join(config.STATISTICS_DIR, "cluster_assignments.csv")

    for p in [meta_path, assign_path]:
        if not os.path.exists(p):
            log.error(f"Required file missing: {p}")
            sys.exit(1)

    df_meta   = pd.read_csv(meta_path)
    df_assign = pd.read_csv(assign_path)

    df = df_assign.merge(df_meta[["image_id", "scene_group"]], on="image_id", how="left")
    df["scene_group"] = df["scene_group"].fillna("unknown")

    log.info(f"Merged {len(df)} rows  |  clusters: {sorted(df['cluster_id'].unique())}")

    # ── Images that have real labels (exclude unknown) ────────────────────────
    df_known = df[df["scene_group"] != "unknown"].copy()
    log.info(f"Images with real scene labels: {len(df_known)} / {len(df)}")

    # ── Per-cluster purity ────────────────────────────────────────────────────
    scene_groups = sorted(df["scene_group"].unique())
    clusters     = sorted(df["cluster_id"].unique())

    purity_rows = []
    for c in clusters:
        c_df = df[df["cluster_id"] == c]
        c_known = c_df[c_df["scene_group"] != "unknown"]
        total   = len(c_df)
        labeled = len(c_known)

        if labeled == 0:
            dominant = "unknown"
            dom_frac = 0.0
            pure_frac = 0.0
        else:
            vc = c_known["scene_group"].value_counts()
            dominant  = vc.idxmax()
            dom_count = int(vc.max())
            dom_frac  = dom_count / labeled
            pure_frac = dom_count / total      # purity against cluster size

        # Per-scene composition (fraction of cluster)
        comp = {sg: 0.0 for sg in scene_groups}
        for sg, cnt in c_df["scene_group"].value_counts().items():
            comp[sg] = round(cnt / total, 4)

        row = {
            "cluster_id"      : int(c),
            "size"            : total,
            "labeled"         : labeled,
            "dominant_scene"  : dominant,
            "dominant_frac_labeled": round(dom_frac, 4),
            "purity"          : round(pure_frac, 4),
        }
        row.update({f"frac_{sg}": comp[sg] for sg in scene_groups})
        purity_rows.append(row)

        log.info(f"  Cluster {c:2d} | size={total:3d} | dominant={dominant:12s} "
                 f"| dom_frac={dom_frac:.3f} | purity={pure_frac:.3f}")

    purity_df = pd.DataFrame(purity_rows)
    purity_path = os.path.join(config.STATISTICS_DIR, "cluster_purity.csv")
    purity_df.to_csv(purity_path, index=False)
    log.info(f"cluster_purity.csv saved: {purity_path}")

    # ── Headline metrics ──────────────────────────────────────────────────────
    # Overall purity = Σ max_label_count_per_cluster / N (on labeled subset)
    overall_purity = sum(
        int(df_known[df_known["cluster_id"] == c]["scene_group"].value_counts().max())
        for c in clusters
        if len(df_known[df_known["cluster_id"] == c]) > 0
    ) / max(len(df_known), 1)

    y_true = df_known["scene_group"].tolist()
    y_pred = df_known["cluster_id"].tolist()
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    hom, comp_score, v_meas = homogeneity_completeness_v_measure(y_true, y_pred)

    summary = {
        "model"             : config.MODEL_NAME,
        "n_images_total"    : int(len(df)),
        "n_images_labeled"  : int(len(df_known)),
        "n_clusters"        : int(config.N_CLUSTERS),
        "overall_purity"    : round(overall_purity, 4),
        "nmi"               : round(nmi, 4),
        "ari"               : round(ari, 4),
        "homogeneity"       : round(hom, 4),
        "completeness"      : round(comp_score, 4),
        "v_measure"         : round(v_meas, 4),
    }

    log.info("\n── Headline Purity Metrics ──")
    for k, v in summary.items():
        log.info(f"  {k:30s}: {v}")

    summary_path = os.path.join(config.STATISTICS_DIR, "purity_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    log.info(f"purity_summary.json saved: {summary_path}")

    # ── Confusion heatmap (cluster × scene_group) ─────────────────────────────
    log.info("Generating confusion heatmap...")
    conf_matrix = pd.crosstab(
        df["cluster_id"], df["scene_group"],
        rownames=["Cluster"], colnames=["Scene Group"]
    )
    # reorder columns
    ordered_cols = [sg for sg in ["road_scene","sky_scene","built_env","green_space","unknown","mixed"]
                    if sg in conf_matrix.columns]
    conf_matrix = conf_matrix[ordered_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_cols) * 1.4), max(6, len(clusters) * 0.8)))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="#DDDDDD",
        cbar_kws={"label": "Image Count"},
        ax=ax,
    )
    ax.set_title(f"Cluster × Scene Group Confusion Matrix\n"
                 f"({config.MODEL_NAME}  |  Purity={overall_purity:.3f}  NMI={nmi:.3f}  ARI={ari:.3f})",
                 fontsize=13, pad=12)
    ax.set_xlabel("Scene Group (from segmentation masks)", fontsize=11)
    ax.set_ylabel("KMeans Cluster ID", fontsize=11)
    plt.tight_layout()
    hm_path = os.path.join(config.FIGURE_DIR, "cluster_purity_heatmap.png")
    plt.savefig(hm_path, dpi=300)
    plt.close()
    log.info(f"Saved: {hm_path}")

    # ── Stacked bar: scene composition per cluster ────────────────────────────
    log.info("Generating stacked bar chart...")
    frac_cols = [f"frac_{sg}" for sg in ordered_cols if f"frac_{sg}" in purity_df.columns]
    bar_data = purity_df.set_index("cluster_id")[frac_cols]
    bar_data.columns = [c.replace("frac_", "") for c in frac_cols]

    bar_colors = [SCENE_COLORS.get(c, "#AAAAAA") for c in bar_data.columns]

    fig, ax = plt.subplots(figsize=(max(10, len(clusters) * 1.2), 6))
    bar_data.plot(kind="bar", stacked=True, color=bar_colors, edgecolor="white",
                  linewidth=0.4, ax=ax, width=0.75)
    ax.set_title(f"Scene Group Composition per Cluster\n"
                 f"({config.MODEL_NAME}  |  Overall Purity = {overall_purity:.3f})",
                 fontsize=13)
    ax.set_xlabel("Cluster ID", fontsize=11)
    ax.set_ylabel("Fraction of Cluster", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(title="Scene Group", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    bar_path = os.path.join(config.FIGURE_DIR, "cluster_scene_composition.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {bar_path}")

    log.info("=" * 60)
    log.info(f"STAGE 10 COMPLETE  Purity={overall_purity:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
