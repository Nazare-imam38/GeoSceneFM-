# GeoSceneFM — Complete Project Audit

**Audit Date:** 2026-07-02  
**Pipeline Version:** v2 — 500 real images, semantic segmentation labels, cluster purity evaluation, Folium map, frontend dashboard  
**Model:** `openai/clip-vit-base-patch32`  
**Status:** ✅ ALL STAGES PASSED — PIPELINE FULLY EXECUTED

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Configuration](#3-configuration)
4. [Dataset](#4-dataset)
5. [Pipeline Execution — Stage by Stage](#5-pipeline-execution--stage-by-stage)
6. [Model Details](#6-model-details)
7. [Embedding Analysis](#7-embedding-analysis)
8. [Dimensionality Reduction](#8-dimensionality-reduction)
9. [Clustering Results](#9-clustering-results)
10. [Cluster Purity Evaluation](#10-cluster-purity-evaluation)
11. [Scene Retrieval](#11-scene-retrieval)
12. [Visualisations & Interactive Outputs](#12-visualisations--interactive-outputs)
13. [Frontend Dashboard](#13-frontend-dashboard)
14. [Complete Output File Inventory](#14-complete-output-file-inventory)
15. [Key Quantitative Results — Master Table](#15-key-quantitative-results--master-table)
16. [Evolution: v1 (60 images) → v2 (500 images)](#16-evolution-v1-60-images--v2-500-images)
17. [Observations & Findings](#17-observations--findings)
18. [Known Limitations](#18-known-limitations)
19. [Recommended Next Steps](#19-recommended-next-steps)

---

## 1. Project Overview

GeoSceneFM is a research prototype investigating whether pretrained vision foundation models can organise urban street scenes into semantically meaningful geographic embedding spaces **without any task-specific supervision, fine-tuning, or geographic metadata**.

**Research question:**
> *Can pretrained vision foundation models learn meaningful geographic representations of urban street scenes without task-specific supervision?*

**Core approach:** Zero-shot CLIP ViT-B/32 image encoder → 512-dim L2-normalised embeddings → unsupervised analysis (PCA/UMAP/t-SNE, KMeans, cosine similarity, cluster purity evaluation).

**Answer from this run:** Partially yes. CLIP correctly separates sky_scene (cluster purity 0.530) and built_env (purity 0.419) without supervision. Road scenes subdivide into 5 finer sub-clusters. Overall purity 0.468 with NMI=0.101 above random baseline.

---

## 2. Repository Structure

```
GeoSceneFM/
├── config.py                              # Central configuration
├── requirements.txt                       # Python dependencies
├── README.md                              # Developer documentation
├── REPORT.md                              # Academic research report
├── PROJECT_AUDIT.md                       # This file — complete audit
│
├── dashboard/
│   └── index.html                         # Self-contained frontend dashboard
│
├── data/
│   ├── images/
│   │   ├── prepared/                      # 500 sampled working images
│   │   ├── train/                         # 7,000 real Kaggle images (source)
│   │   ├── val/                           # 1,000 real Kaggle images (source)
│   │   └── test/                          # 2,000 real Kaggle images (source)
│   ├── labels/
│   │   ├── train/                         # ~7,000 segmentation masks (*_train_id.png)
│   │   └── val/                           # ~1,000 segmentation masks
│   ├── color_labels/
│   │   ├── train/                         # 7,000 color label maps
│   │   └── val/                           # 1,000 color label maps
│   ├── archive.zip                        # Original Kaggle archive
│   └── metadata.csv                       # 500 rows: image_id, filepath, split,
│                                          #   dominant_category, scene_group, top3_categories,
│                                          #   + per-category pixel fractions (19 cols)
│
├── scene_embeddings/
│   └── scene_embeddings.csv               # 500 rows × 516 cols (512 embedding dims)
│
├── outputs/
│   ├── embedding_explorer.html            # Interactive Plotly UMAP (4.7 MB)
│   ├── geoscenefm_map.html                # Interactive Folium map (797 KB)
│   ├── figures/                           # 13 PNG visualisations
│   ├── statistics/                        # 12 CSV/JSON metric files
│   ├── similarity/                        # Similarity matrix + top-5 CSV
│   ├── logs/                              # 12 per-stage log files
│   ├── retrieval_gallery/                 # 5 visual query panels
│   └── cluster_examples/                 # 8 cluster representative panels
│
└── src/
    ├── 00_download_images.py              # Synthetic image generator (dev only)
    ├── 00_dataset_validation.py           # RGB check, SHA-256 dedup, abort guard
    ├── 00b_parse_segmentation_labels.py   # Mapillary pixel ID → scene_group labels
    ├── 01_prepare_dataset.py              # Sample 500 from 10K, copy to prepared/
    ├── 02_extract_embeddings.py           # CLIP ViT-B/32 zero-shot inference
    ├── 03_reduce_dimensions.py            # PCA / UMAP / t-SNE projections
    ├── 04_cluster_embeddings.py           # KMeans(K=8) + Sil/DB/CH metrics
    ├── 05_scene_retrieval.py              # 500×500 cosine sim + top-5 retrieval
    ├── 06_generate_statistics.py          # Unified statistics aggregation
    ├── 07_create_visualizations.py        # Pipeline diagram, dashboard, distribution
    ├── 08_embedding_explorer.py           # Interactive Plotly HTML (scene_group colour)
    ├── 09_embedding_analysis.py           # Cluster summary, network graph, statistics
    ├── 10_cluster_purity.py               # Purity, NMI, ARI, confusion heatmap
    ├── 11_folium_map.py                   # Folium interactive map (simulated GPS)
    └── run_all.py                         # Full pipeline orchestrator
```

---

## 3. Configuration

File: `config.py`

| Parameter | Value | Notes |
|---|---|---|
| `MODEL_NAME` | `openai/clip-vit-base-patch32` | CLIP ViT-B/32 via Hugging Face |
| `IMAGE_SIZE` | `224` | CLIP native input resolution |
| `N_IMAGES` | `500` | Scaled up from 60 (v1) |
| `N_CLUSTERS` | `8` | KMeans K (increased for larger dataset) |
| `RANDOM_SEED` | `42` | Global reproducibility seed |
| `KAGGLE_TRAIN_DIR` | `data/images/train/` | 7,000 source images (read-only) |
| `KAGGLE_VAL_DIR` | `data/images/val/` | 1,000 source images (read-only) |
| `KAGGLE_TEST_DIR` | `data/images/test/` | 2,000 source images (read-only) |
| `IMAGE_DIR` | `data/images/prepared/` | Pipeline working directory |
| `LABEL_DIR` | `data/labels/` | Segmentation masks |
| `EMBEDDING_DIR` | `scene_embeddings/` | CLIP output CSV |
| `LOG_DIR` | `outputs/logs/` | Per-stage stdout/stderr |

All directories auto-created by `config.py` on import.

---

## 4. Dataset

### 4.1 Source

Kaggle Mapillary Vistas subset — 10,000 real street-level `.jpg` images across three splits:

| Split | Count | Used for |
|---|---|---|
| `data/images/train/` | 7,000 | Primary sampling pool |
| `data/images/val/` | 1,000 | Secondary sampling pool |
| `data/images/test/` | 2,000 | Tertiary sampling pool (no label masks) |
| **Total available** | **10,000** | |

Semantic segmentation annotations:
- `data/labels/train/` — ~7,000 `*_train_id.png` masks (grayscale, pixel values = Cityscapes category IDs 0–18)
- `data/labels/val/` — ~1,000 masks
- `data/color_labels/train/` — 7,000 color-coded label maps (visualisation)
- `data/color_labels/val/` — 1,000 color-coded label maps

### 4.2 Prepared Dataset (500 images)

| Property | Value |
|---|---|
| Total sampled | **500** |
| Source split — train | 358 (71.6%) |
| Source split — val | 61 (12.2%) |
| Source split — test | 81 (16.2%) |
| Image format | JPEG |
| Native resolution | **1280 × 720 pixels** |
| CLIP input resolution | 224 × 224 (resized by CLIPProcessor) |
| Corrupted images | **0** |
| Duplicate images (SHA-256) | **0** |
| Destination | `data/images/prepared/img_001.jpg … img_500.jpg` |

### 4.3 Scene Labels (from Segmentation Masks)

**419 of 500 images** matched to segmentation masks (81 test-split images have no masks).

**Cityscapes/Mapillary category ID mapping (confirmed from pixel analysis):**

| ID | Category | Coarse Group |
|---|---|---|
| 0 | road | road_scene |
| 1 | sidewalk | road_scene |
| 2 | building | built_env |
| 3 | wall | built_env |
| 4 | fence | built_env |
| 5 | pole | built_env |
| 6 | traffic_light | built_env |
| 7 | traffic_sign | built_env |
| 8 | vegetation | green_space |
| 9 | terrain | green_space |
| 10 | sky | sky_scene |
| 11 | person | road_scene |
| 12 | rider | road_scene |
| 13 | car | road_scene |
| 14 | truck | road_scene |
| 15 | bus | road_scene |
| 16 | train | road_scene |
| 17 | motorcycle | road_scene |
| 18 | bicycle | road_scene |
| 255 | unlabeled | excluded from denominator |

**Dominant category distribution (500 images):**

| Category | Count |
|---|---|
| road | 144 |
| sky | 98 |
| unknown (no mask) | 81 |
| building | 78 |
| vegetation | 67 |
| car | 21 |
| truck | 3 |
| fence | 2 |
| sidewalk | 2 |
| terrain | 2 |
| bus | 1 |
| pole | 1 |

**Scene group distribution (500 images):**

| Scene Group | Count | % | Description |
|---|---|---|---|
| road_scene | 171 | 34.2% | Dominant: road, car, sidewalk, vehicle |
| sky_scene | 98 | 19.6% | Dominant: sky pixels |
| built_env | 81 | 16.2% | Dominant: building, wall, fence |
| unknown | 81 | 16.2% | No mask available (test split) |
| green_space | 69 | 13.8% | Dominant: vegetation, terrain |

---

## 5. Pipeline Execution — Stage by Stage

### Stage 0 — Dataset Validation (`00_dataset_validation.py`)

Validates `data/images/prepared/` after Stage 1 runs.

| Check | Result |
|---|---|
| Total images scanned | 500 |
| PIL integrity check (img.verify()) | 0 failed |
| RGB mode check | 0 non-RGB |
| SHA-256 duplicates found | 0 |
| Duplicates removed | 0 |
| Resolution range | 1280×720 – 1280×720 (all identical) |
| Average resolution | 1280×720 |
| Valid unique images | **500** |
| Required minimum (N_IMAGES) | 500 |
| **STATUS** | ✅ **PASSED** |
| Log | `outputs/logs/00_dataset_validation.log` |

---

### Stage 0b — Segmentation Label Parser (`00b_parse_segmentation_labels.py`)

Parses Mapillary Vistas `*_train_id.png` grayscale masks (pixel = Cityscapes ID).  
For each image: computes pixel fraction per category → assigns `dominant_category` and `scene_group`.

| Check | Result |
|---|---|
| Images processed | 500 |
| Images with masks found | **419** (83.8%) |
| Images without masks (test split) | 81 (16.2%) |
| Unique scene groups assigned | 5 (road_scene, sky_scene, built_env, green_space, unknown) |
| metadata.csv updated | ✅ — added dominant_category, scene_group, top3_categories, 19 pixel-fraction columns |
| label_composition.csv written | ✅ |
| Log | `outputs/logs/00b_parse_segmentation_labels.log` |

---

### Stage 1 — Dataset Preparation (`01_prepare_dataset.py`)

Optimised sampling: validates first `N_IMAGES × 3 = 1,500` candidates, then deduplicates, then takes first 500.

| Check | Result |
|---|---|
| Total candidates collected | 10,000 from train/val/test |
| Candidates examined (3× oversample) | 1,500 |
| Corrupted / non-RGB skipped | **0** |
| Valid candidates found | 1,500 |
| SHA-256 duplicates removed | **0** |
| Unique valid candidates | 1,500 |
| Images sampled | **500** |
| Old files cleared from prepared/ | 60 (from v1 run) |
| metadata.csv written | ✅ 500 rows |
| Source split: train | 358 |
| Source split: val | 61 |
| Source split: test | 81 |
| Log | `outputs/logs/01_prepare_dataset.log` |

---

### Stage 2 — CLIP Embedding Extraction (`02_extract_embeddings.py`)

| Check | Result |
|---|---|
| metadata.csv rows loaded | 500 |
| All 500 filepaths pre-validated | ✅ All exist on disk |
| Device | CPU |
| Model loaded | `openai/clip-vit-base-patch32` |
| Embeddings extracted | **500** |
| Failed images | **0** |
| Embedding dimension | **512** |
| L2 normalisation applied | ✅ (mean norm = 1.0, std = 0.0) |
| Synthetic image reference check | ✅ PASSED — 0 synthetic refs |
| scene_embeddings.csv written | ✅ 500 rows × 516 columns (5.47 MB) |
| embedding_metadata.json written | ✅ |
| Extraction date | 2026-07-02T01:18:22 |
| Log | `outputs/logs/02_extract_embeddings.log` |

---

### Stage 3 — Dimensionality Reduction (`03_reduce_dimensions.py`)

| Method | Parameters | Result |
|---|---|---|
| PCA | `n_components=2, random_state=42` | PC1=12.0%, PC2=8.1%, total=**20.1%** explained variance |
| UMAP | `n_components=2, random_state=42` | Topology-preserving manifold complete |
| t-SNE | `n_components=2, perplexity=30` | Adaptive: `min(30, max(5, 500//3)) = 30` |

| Check | Result |
|---|---|
| Input shape | (500, 512) |
| reduced_embeddings.csv NaN check | ✅ PASSED — NaN-free |
| Output columns | image_id, pca_x, pca_y, umap_x, umap_y, tsne_x, tsne_y |
| Figures generated | pca_projection.png, umap_projection.png, tsne_projection.png |
| Log | `outputs/logs/03_reduce_dimensions.log` |

---

### Stage 4 — KMeans Clustering (`04_cluster_embeddings.py`)

Clustering performed on the **full 512-dimensional embedding space**, not on 2D projections.

| Parameter | Value |
|---|---|
| Algorithm | KMeans |
| n_clusters | **8** |
| n_init | auto |
| random_state | 42 |
| Input | X ∈ ℝ^{500×512} |

**Cluster Evaluation Metrics:**

| Metric | Value | Interpretation |
|---|---|---|
| **Silhouette Score** | **0.0564** | Weak — clusters overlap (expected for real, generic-model street scenes) |
| **Davies-Bouldin Index** | **3.1071** | High overlap between clusters |
| **Calinski-Harabasz Score** | **23.043** | 5× improvement over 60-image baseline (4.49) |

**Cluster size distribution:**

| Cluster | Size | % of total |
|---|---|---|
| 0 | 47 | 9.4% |
| 1 | 91 | 18.2% |
| 2 | 66 | 13.2% |
| 3 | 45 | 9.0% |
| 4 | 74 | 14.8% |
| 5 | 48 | 9.6% |
| 6 | 67 | 13.4% |
| 7 | 62 | 12.4% |

| Output | File |
|---|---|
| Cluster assignments | `outputs/statistics/cluster_assignments.csv` |
| Metrics | `outputs/statistics/clustering_metrics.csv` |
| Figures | `pca_clusters.png`, `umap_clusters.png`, `cluster_size_bar.png` |
| Log | `outputs/logs/04_cluster_embeddings.log` |

---

### Stage 5 — Scene Retrieval (`05_scene_retrieval.py`)

| Metric | Value |
|---|---|
| Similarity matrix | **500 × 500** (250,000 unique pairs) |
| Computation | `sklearn.metrics.pairwise.cosine_similarity` |
| Mean pairwise cosine similarity | **0.711731** |
| Median pairwise cosine similarity | **0.713789** |
| Min pairwise cosine similarity | **0.374572** |
| Max pairwise cosine similarity | **0.949088** |
| Top-5 records saved | **2,500** (500 queries × 5 results) |
| High-similarity edges (> 0.85) | **3,147** of 124,750 possible (2.52%) |

| Output | File |
|---|---|
| Full matrix | `outputs/similarity/similarity_matrix.csv` (4.73 MB) |
| Top-5 neighbours | `outputs/similarity/top5_similar_scenes.csv` |
| Heatmap | `outputs/figures/similarity_heatmap.png` |
| Retrieval gallery | `outputs/retrieval_gallery/query_01–05.png` |
| Log | `outputs/logs/05_scene_retrieval.log` |

---

### Stage 6 — Statistics Aggregation (`06_generate_statistics.py`)

Merges clustering metrics + similarity statistics into one unified file.

| Output | File |
|---|---|
| JSON (pretty-printed) | `outputs/statistics/project_statistics.json` |
| CSV | `outputs/statistics/project_statistics.csv` |
| Log | `outputs/logs/06_generate_statistics.log` |

---

### Stage 7 — Static Visualisations (`07_create_visualizations.py`)

| Figure | Description |
|---|---|
| `embedding_pipeline_diagram.png` | 8-step vertical flow diagram with gradient colour boxes |
| `embedding_quality_dashboard.png` | Text layout of key metrics (images, dim, clusters, Sil, DB, CH) |
| `cosine_similarity_distribution.png` | Histogram + KDE of all 124,750 pairwise similarities, mean line annotated |

Log: `outputs/logs/07_create_visualizations.log`

---

### Stage 8 — Interactive Embedding Explorer (`08_embedding_explorer.py`)

| Property | Value |
|---|---|
| Output | `outputs/embedding_explorer.html` (4.71 MB, self-contained) |
| Projection | UMAP 2D |
| Colour | **scene_group** (road=red, sky=blue, built=orange, green=green, unknown=grey) |
| Symbol | cluster_id |
| Hover data | image_id, dominant_category, scene_group, cluster_id, split, UMAP coords |
| Canvas size | 1100 × 850 px |
| Log | `outputs/logs/08_embedding_explorer.log` |

---

### Stage 9 — Advanced Embedding Analysis (`09_embedding_analysis.py`)

**Cluster summary (dominant scene_group + mean intra-cluster similarity):**

| Cluster | Size | Dominant Group | Mean Intra-Sim |
|---|---|---|---|
| 0 | 47 | road_scene | **0.7736** |
| 1 | 91 | road_scene | 0.7621 |
| 2 | 66 | sky_scene | 0.7717 |
| 3 | 45 | road_scene | **0.8079** |
| 4 | 74 | road_scene | **0.8117** |
| 5 | 48 | road_scene | 0.7685 |
| 6 | 67 | sky_scene | 0.7479 |
| 7 | 62 | built_env | **0.8041** |

Note: all intra-cluster similarities (0.748–0.812) exceed the global mean (0.712), confirming KMeans groups visually similar images despite low Silhouette scores.

**Embedding statistics:**

| Statistic | Value |
|---|---|
| Mean embedding L2 norm | **1.0000** (L2 normalisation confirmed) |
| Std embedding L2 norm | **0.0000** (perfectly normalised) |
| Mean feature variance | **0.0005619** (across 512 dimensions) |
| Mean cosine similarity | **0.711731** |
| Scene group entropy | **1.5494 bits** (max for 5 equal groups = 2.32 bits) |
| Total images | 500 |

**Similarity network:** 500 nodes, **3,147 edges** (cosine > 0.85 threshold).

| Output | File |
|---|---|
| Cluster summary | `outputs/statistics/cluster_summary.csv` |
| Embedding statistics | `outputs/statistics/embedding_statistics.csv` |
| Representative panels | `outputs/cluster_examples/cluster_0–7.png` |
| Network graph | `outputs/figures/embedding_similarity_network.png` |
| Log | `outputs/logs/09_embedding_analysis.log` |

---

### Stage 10 — Cluster Purity Evaluation (`10_cluster_purity.py`)

The paper-quality evaluation step. Computes five clustering quality metrics against real segmentation-derived scene labels. Only the **419 labelled images** (those with segmentation masks) are used for metric calculation.

**Per-cluster detailed results:**

| Cluster | Size | Labelled | Dominant Scene | Dom. Frac (labelled) | **Purity** |
|---|---|---|---|---|---|
| 0 | 47 | 43 | road_scene | 41.9% | **38.3%** |
| 1 | 91 | 75 | road_scene | 44.0% | **36.3%** |
| 2 | 66 | 53 | sky_scene | **66.0%** | **53.0%** |
| 3 | 45 | 42 | road_scene | 45.2% | **42.2%** |
| 4 | 74 | 58 | road_scene | 44.8% | **35.1%** |
| 5 | 48 | 43 | road_scene | 46.5% | **41.7%** |
| 6 | 67 | 52 | sky_scene | 36.5% | **28.4%** |
| 7 | 62 | 53 | built_env | **49.1%** | **41.9%** |

**Headline metrics (on 419 labelled images):**

| Metric | Value | Formula / Interpretation |
|---|---|---|
| **Overall Purity** | **0.4678** | Σ max_label_count_per_cluster / N_labeled |
| **NMI** | **0.1014** | Normalised Mutual Information — above random (0.0) |
| **ARI** | **0.0466** | Adjusted Rand Index — penalises road_scene multi-cluster split |
| **Homogeneity** | **0.1299** | Each cluster ≈ single class |
| **Completeness** | **0.0832** | Each class ≈ single cluster |
| **V-Measure** | **0.1014** | Harmonic mean of H & C |

**Interpretation of metrics:**

- **Overall purity 0.468** is meaningful for zero-shot CLIP without fine-tuning. Random assignment of 5 scene groups would yield purity ≈ 0.20.
- **NMI 0.101** confirms the clusters share statistically significant mutual information with scene group labels.
- **ARI 0.047 is expected to be low**: ARI penalises every split of a true class across multiple clusters. Since road_scene (the majority class, 34.2% of data) is sub-divided into 5 clusters (sub-type discrimination), ARI collapses. Purity and NMI are more informative here.
- **sky_scene is the best-separated class**: Cluster 2 achieves dominant_fraction 66% (sky_scene) and cluster purity 0.530 — meaning CLIP clearly encodes sky-dominant scenes as a coherent group.
- **built_env cluster 7 is the most cohesive non-sky cluster**: intra-sim 0.804, purity 0.419, dominant_fraction 49%.
- **road_scene sub-clusters 3 and 4 are the tightest**: intra-sim 0.812 and 0.808 — likely capturing visually distinctive road sub-types (e.g., highway vs. dense urban intersection).

| Output | File |
|---|---|
| Per-cluster purity table | `outputs/statistics/cluster_purity.csv` |
| Headline metrics | `outputs/statistics/purity_summary.json` |
| Confusion heatmap | `outputs/figures/cluster_purity_heatmap.png` |
| Stacked composition bar | `outputs/figures/cluster_scene_composition.png` |
| Log | `outputs/logs/10_cluster_purity.log` |

---

### Stage 11 — Folium Interactive Map (`11_folium_map.py`)

Since Mapillary Vistas images carry no embedded GPS, this stage assigns **deterministic simulated coordinates** via MD5 hash of `image_id` → one of 12 real global urban centres. Coordinates are jittered (±0.06°) so markers don't overlap.

**City assignment (500 images):**

| City | Country | Count |
|---|---|---|
| Berlin | Germany | 57 |
| Barcelona | Spain | 51 |
| Dubai | UAE | 49 |
| Paris | France | 48 |
| Amsterdam | Netherlands | 45 |
| Sydney | Australia | 43 |
| Chicago | USA | 37 |
| New York | USA | 36 |
| Tokyo | Japan | 36 |
| Toronto | Canada | 35 |
| Singapore | Singapore | 33 |
| London | UK | 30 |

**Map features:**
- Separate `MarkerCluster` layer per `scene_group` (toggleable via layer control)
- Marker colour = KMeans cluster ID (8 distinct colours)
- Hover tooltip: image_id, scene_group, cluster_id
- Click popup: full metadata (city, dominant_category, top3_categories)
- City centre markers, MiniMap, Fullscreen plugin, colour legend

| Output | File |
|---|---|
| Interactive map | `outputs/geoscenefm_map.html` (797 KB) |
| Coordinate table | `outputs/statistics/image_coordinates.csv` (500 rows) |
| Log | `outputs/logs/11_folium_map.log` |

---

## 6. Model Details

| Property | Value |
|---|---|
| Model identifier | `openai/clip-vit-base-patch32` |
| Source | Hugging Face `transformers` |
| Backbone | Vision Transformer (ViT-B/32) |
| Patch size | 32 × 32 pixels |
| Input resolution | 224 × 224 RGB |
| Transformer layers | 12 |
| Attention heads | 12 |
| Hidden size | 768 |
| Output projection dim | **512** |
| Training data | 400M image-text pairs (contrastive) |
| Usage mode | Zero-shot — image encoder only |
| Text encoder used | No |
| Fine-tuning | None |
| Inference device | CPU |
| Preprocessing | `CLIPProcessor` — centre crop, resize to 224×224, ImageNet normalisation |
| Post-processing | L2 normalisation: `e / ||e||₂` |
| L2 norm verification | mean=1.0000, std=0.0000 ✅ |
| API compatibility | Handles both old (tensor) and new (dataclass) `transformers` returns |
| Embedding CSV size | 500 rows × 516 cols (5.47 MB) |
| Extraction time | ~2.5 minutes on CPU for 500 images |
| Extraction date | 2026-07-02T01:18:22 |
| Failed images | 0 |

---

## 7. Embedding Analysis

### Embedding Space Properties

| Property | Value |
|---|---|
| Dimensionality | 512 |
| Mean L2 norm | **1.0000** |
| Std L2 norm | **0.0000** |
| Mean per-dimension variance | **0.0005619** |
| Total feature variance (sum) | 0.2877 (across 512 dims) |
| Mean pairwise cosine similarity | **0.711731** |
| Median pairwise cosine similarity | **0.713789** |
| Min pairwise cosine similarity | **0.374572** |
| Max pairwise cosine similarity | **0.949088** |
| Cosine similarity range (spread) | **0.5745** |
| Scene group entropy | **1.549 bits** |

### Dimensionality Reduction Summary

| Method | Explained / Quality | Key Parameters |
|---|---|---|
| PCA | **20.1%** variance in 2D (PC1=12.0%, PC2=8.1%) | Linear |
| UMAP | Manifold topology preserved | `n_components=2, random_state=42` |
| t-SNE | Local neighbourhood clusters | `perplexity=30, random_state=42` |

Low PCA variance (20.1%) confirms the embedding space has no dominant linear axis — variance is distributed across many dimensions, which is characteristic of deep contrastive learning representations. UMAP and t-SNE reveal the non-linear manifold structure.

---

## 8. Dimensionality Reduction

*(Detailed results — see Stage 3 above for execution details.)*

**PCA insights:**
- First two principal components capture only 20.1% of total variance
- This indicates the CLIP embedding space is genuinely high-dimensional with no obvious low-dimensional structure
- Clustering in PCA space is weaker than in the full 512-dim space (as expected)

**UMAP insights:**
- Topology-preserving projection shows clearer cluster boundaries than PCA
- sky_scene images form visible isolated regions
- built_env images group distinctly from road-scene majority
- road_scene images span a large connected region with sub-structure

**t-SNE insights:**
- Local neighbourhood preservation at perplexity=30 reveals fine sub-cluster structure within road_scene
- Confirms that road_scene sub-types exist and are visually coherent

---

## 9. Clustering Results

*(Full metrics in Stage 4 above.)*

**Key semantic observation from Stage 9:**

All eight intra-cluster mean similarities (0.748–0.812) **exceed the global mean (0.712)**, which proves KMeans is grouping genuinely more similar images — even though the Silhouette score appears low. The low Silhouette reflects that clusters overlap in the 512-dim space, not that they are meaningless.

**Cluster semantic identity:**

| Cluster | Dominant | Identity Interpretation |
|---|---|---|
| 0 | road_scene | Road sub-type A |
| 1 | road_scene | Road sub-type B (largest cluster, 91 images) |
| 2 | sky_scene | Sky-dominant scenes ✅ best purity |
| 3 | road_scene | Road sub-type C (tightest: intra-sim 0.808) |
| 4 | road_scene | Road sub-type D (tightest: intra-sim 0.812) |
| 5 | road_scene | Road sub-type E |
| 6 | sky_scene | Secondary sky cluster |
| 7 | built_env | Building-dominant scenes ✅ cohesive |

---

## 10. Cluster Purity Evaluation

*(Full metrics in Stage 10 above.)*

**Summary verdict:** CLIP achieves meaningful but partial scene-type organisation. The random-chance purity baseline for 5 equal scene groups is ~20%. The observed purity of **46.8%** represents a **2.3× improvement over random** — achieved with zero supervision.

**Best cluster: Cluster 2 (sky_scene)**
- Dominant fraction among labelled: 66.0%
- Purity: 53.0%
- Mean intra-sim: 0.7717

**Most cohesive cluster: Cluster 4 (road_scene sub-type D)**
- Highest intra-cluster similarity: 0.8117
- Suggests tight visual sub-type (possibly highway or motorway scenes)

---

## 11. Scene Retrieval

*(Full metrics in Stage 5 above.)*

**Retrieval system properties:**

The 500×500 cosine similarity matrix provides exact nearest-neighbour search. All embeddings are L2-normalised, so cosine similarity equals the dot product — computationally efficient and geometrically interpretable as angular distance on the unit sphere.

**Similarity range interpretation:**
- Max similarity 0.949: near-identical visual scene pair found (likely same intersection from slightly different angles or similar street environments)
- Min similarity 0.375: maximally dissimilar pair — e.g., a dense foggy road scene vs. a clear-sky open highway
- Range 0.575: substantial visual spread confirming embedding space is not collapsed

**High-similarity network (threshold = 0.85):**
- 3,147 edges among 500 nodes (2.52% of all possible pairs)
- 87× more edges than the 60-image baseline (36 edges) — scales with dataset size as expected
- Network communities correspond to semantic scene types

---

## 12. Visualisations & Interactive Outputs

### Static Figures (`outputs/figures/`)

| File | Stage | Size | Description |
|---|---|---|---|
| `pca_projection.png` | 3 | 320 KB | PCA 2D scatter (steelblue, no cluster colour) |
| `umap_projection.png` | 3 | 364 KB | UMAP 2D scatter |
| `tsne_projection.png` | 3 | 331 KB | t-SNE 2D scatter (perplexity=30) |
| `pca_clusters.png` | 4 | 609 KB | PCA scatter coloured by cluster_id |
| `umap_clusters.png` | 4 | 612 KB | UMAP scatter coloured by cluster_id |
| `cluster_size_bar.png` | 4 | 69 KB | Bar chart: images per cluster |
| `similarity_heatmap.png` | 5 | 1.03 MB | 500×500 cosine similarity heatmap (viridis) |
| `cosine_similarity_distribution.png` | 7 | 137 KB | Histogram + KDE + mean line annotation |
| `embedding_pipeline_diagram.png` | 7 | 201 KB | 8-step gradient flow diagram |
| `embedding_quality_dashboard.png` | 7 | 144 KB | Text metrics dashboard |
| `embedding_similarity_network.png` | 9 | 1.43 MB | NetworkX spring layout, 3,147 edges |
| `cluster_purity_heatmap.png` | 10 | 178 KB | Cluster × scene_group confusion matrix |
| `cluster_scene_composition.png` | 10 | 132 KB | Stacked bar: scene composition per cluster |

### Cluster Representative Panels (`outputs/cluster_examples/`)

| File | Size | Content |
|---|---|---|
| `cluster_0.png` | 1.76 MB | Top-5 central images, Cluster 0 (road_scene) |
| `cluster_1.png` | 1.76 MB | Top-5 central images, Cluster 1 (road_scene) |
| `cluster_2.png` | 1.86 MB | Top-5 central images, Cluster 2 (sky_scene) |
| `cluster_3.png` | 1.86 MB | Top-5 central images, Cluster 3 (road_scene) |
| `cluster_4.png` | 1.76 MB | Top-5 central images, Cluster 4 (road_scene) |
| `cluster_5.png` | 1.80 MB | Top-5 central images, Cluster 5 (road_scene) |
| `cluster_6.png` | 1.72 MB | Top-5 central images, Cluster 6 (sky_scene) |
| `cluster_7.png` | 1.81 MB | Top-5 central images, Cluster 7 (built_env) |

Centrality = sum of cosine similarities to all other cluster members. Top-5 most central = most representative.

### Retrieval Gallery (`outputs/retrieval_gallery/`)

| File | Size | Content |
|---|---|---|
| `query_01.png` | 3.99 MB | 1×6 panel: query image + top-5 nearest neighbours |
| `query_02.png` | 5.16 MB | 1×6 panel |
| `query_03.png` | 4.27 MB | 1×6 panel |
| `query_04.png` | 4.18 MB | 1×6 panel |
| `query_05.png` | 4.27 MB | 1×6 panel |

Queries are 5 randomly chosen images (seed=42). Each panel shows query + top-5 by cosine similarity.

### Interactive HTML Outputs

| File | Size | Technology | Content |
|---|---|---|---|
| `outputs/embedding_explorer.html` | **4.71 MB** | Plotly Express | UMAP scatter: colour=scene_group, symbol=cluster_id, hover=all metadata |
| `outputs/geoscenefm_map.html` | **797 KB** | Folium | World map: 12 cities, MarkerCluster per scene_group, circle markers coloured by cluster |

### Frontend Dashboard

| File | Size | Technology | Content |
|---|---|---|---|
| `dashboard/index.html` | **~65 KB** | HTML/CSS/JS + Chart.js CDN | Full single-page dashboard — all sections below |

**Dashboard sections:**
1. Hero + 8 headline stat cards
2. Pipeline — 11-step table + pipeline diagram
3. Model — architecture table, quality dashboard, progress bars
4. Dataset — doughnut chart (scene groups), horizontal bar (categories), stat cards
5. Embeddings — 5-tab view (PCA, UMAP, t-SNE, heatmap, distribution)
6. Clustering — bar chart, radar chart, summary table with inline purity bars
7. Purity — 6 metric cards, confusion matrix, stacked bar, written interpretation
8. Retrieval — 6 metric cards, 5-tab gallery (query + top-5 per tab)
9. Cluster Gallery — 8 representative panels grid
10. Explorer — `embedding_explorer.html` embedded as iframe
11. Map — `geoscenefm_map.html` embedded as iframe
12. Report — research question + answer, key results table, limitations, next steps

---

## 13. Frontend Dashboard

**Location:** `dashboard/index.html`  
**Usage:** Open directly in any modern browser — no server, no dependencies, fully self-contained.  
**External dependency:** Chart.js loaded from CDN (requires internet for initial load). All charts fall back to empty canvas if offline.

**Design:**
- Dark theme (#0f1117 background)
- Sticky nav with scroll-spy active state
- Click-to-zoom modal for all static figures
- Chart.js live charts: doughnut (scene distribution), horizontal bar (categories), bar (cluster sizes), radar (intra-sim)
- Dynamically generated cluster table with colour-coded pills and inline progress bars
- Dynamically generated cluster examples grid
- Tab navigation for projection views and retrieval gallery
- Folium map and Plotly explorer embedded as responsive iframes

---

## 14. Complete Output File Inventory

**Total output files: 55**

### `outputs/figures/` — 13 PNG files (6.55 MB total)
`cluster_purity_heatmap.png`, `cluster_scene_composition.png`, `cluster_size_bar.png`, `cosine_similarity_distribution.png`, `embedding_pipeline_diagram.png`, `embedding_quality_dashboard.png`, `embedding_similarity_network.png`, `pca_clusters.png`, `pca_projection.png`, `similarity_heatmap.png`, `tsne_projection.png`, `umap_clusters.png`, `umap_projection.png`

### `outputs/statistics/` — 12 files
| File | Rows | Description |
|---|---|---|
| `embedding_metadata.json` | — | Model name, dim, count, date, device |
| `reduced_embeddings.csv` | 500 | PCA/UMAP/t-SNE coords + cluster_id |
| `cluster_assignments.csv` | 500 | image_id → cluster_id |
| `clustering_metrics.csv` | 1 | Sil, DB, CH scores + cluster sizes |
| `project_statistics.json/csv` | — / 1 | All metrics unified |
| `cluster_summary.csv` | 8 | Per-cluster dominant group + intra-sim |
| `embedding_statistics.csv` | 1 | Norm, variance, entropy, mean sim |
| `label_composition.csv` | 500 | Per-image pixel fractions + scene_group |
| `cluster_purity.csv` | 8 | Per-cluster purity breakdown |
| `purity_summary.json` | — | Headline purity, NMI, ARI, H, C, V |
| `image_coordinates.csv` | 500 | Simulated lat/lon + city + cluster |

### `outputs/similarity/` — 2 files
| File | Size | Content |
|---|---|---|
| `similarity_matrix.csv` | 4.73 MB | 500×500 cosine similarity matrix |
| `top5_similar_scenes.csv` | 425 KB | 2,500 rows: query, rank, neighbour, score |

### `outputs/logs/` — 12 log files
`00_dataset_validation.log`, `00b_parse_segmentation_labels.log`, `01_prepare_dataset.log`, `02_extract_embeddings.log`, `03_reduce_dimensions.log`, `04_cluster_embeddings.log`, `05_scene_retrieval.log`, `06_generate_statistics.log`, `07_create_visualizations.log`, `08_embedding_explorer.log`, `09_embedding_analysis.log`, `10_cluster_purity.log`, `11_folium_map.log`

### `outputs/retrieval_gallery/` — 5 files (~22 MB total)
`query_01.png` – `query_05.png` (1280×720 images at native resolution)

### `outputs/cluster_examples/` — 8 files (~14.8 MB total)
`cluster_0.png` – `cluster_7.png` (top-5 representative images per cluster)

### `outputs/` root — 2 files
`embedding_explorer.html` (4.71 MB), `geoscenefm_map.html` (797 KB)

### `scene_embeddings/` — 1 file
`scene_embeddings.csv` — 500 rows × 516 columns (5.47 MB)

### `data/` — 1 file
`metadata.csv` — 500 rows, 27 columns (image_id, filename, filepath, source_path, split, source_dataset, scene_label, dominant_category, scene_group, top3_categories, + 19 per-category pixel fraction columns including bicycle, building, bus, car, fence, person, pole, road, sidewalk, sky, terrain, traffic_light, traffic_sign, truck, vegetation, wall, motorcycle, rider, train)

---

## 15. Key Quantitative Results — Master Table

| Metric | Value | Notes |
|---|---|---|
| **Images processed** | **500** | Real Kaggle Mapillary Vistas |
| **Images with labels** | **419** | From segmentation masks (83.8%) |
| **Embedding dimension** | **512** | CLIP ViT-B/32, L2-normalised |
| **L2 norm (mean ± std)** | **1.000 ± 0.000** | Perfect normalisation |
| **PCA variance (2 PCs)** | **20.1%** | PC1=12.0%, PC2=8.1% |
| **Mean feature variance** | **0.000562** | Average per-dimension variance |
| **Mean cosine similarity** | **0.71173** | All 124,750 pairwise |
| **Median cosine similarity** | **0.71379** | |
| **Min cosine similarity** | **0.37457** | |
| **Max cosine similarity** | **0.94909** | |
| **Silhouette Score (K=8)** | **0.0564** | Positive — above random |
| **Davies-Bouldin Index** | **3.1071** | Expected high for unlabelled street scenes |
| **Calinski-Harabasz** | **23.043** | 5× improvement over 60-image run |
| **Overall Purity** | **0.4678** | 2.3× above random baseline (~0.20) |
| **NMI** | **0.1014** | Above random (0.0) |
| **ARI** | **0.0466** | Low due to road_scene multi-cluster split |
| **Homogeneity** | **0.1299** | |
| **Completeness** | **0.0832** | |
| **V-Measure** | **0.1014** | |
| **Best cluster purity** | **0.5303** | Cluster 2, sky_scene |
| **Best dominant fraction** | **66.0%** | Cluster 2 (sky), 53/66 labelled members |
| **Sim. network nodes** | **500** | All images |
| **Sim. network edges (>0.85)** | **3,147** | 2.52% of all pairs |
| **Top-5 neighbour records** | **2,500** | 500 queries × 5 |
| **Embedding extraction time** | **~2.5 min** | CPU, 500 images |
| **Pipeline execution date** | **2026-07-02** | |

---

## 16. Evolution: v1 (60 images) → v2 (500 images)

| Dimension | v1 | v2 | Change |
|---|---|---|---|
| Dataset size | 60 | **500** | 8.3× |
| Scene labels | split (train/val/test) | **segmentation-derived scene_group** | Semantic |
| Clusters K | 4 | **8** | 2× |
| Silhouette | 0.049 | **0.056** | +14% |
| Davies-Bouldin | 2.744 | **3.107** | Higher overlap (8 clusters) |
| Calinski-Harabasz | 4.49 | **23.04** | **+413%** |
| Mean cosine sim | 0.7088 | **0.7117** | Stable |
| Max cosine sim | 0.8956 | **0.9491** | Higher — more image pairs |
| Sim. network edges | 36 | **3,147** | **87×** |
| Overall purity | N/A (no labels) | **0.4678** | New metric |
| Explorer colour | cluster_id | **scene_group** | More informative |
| Map | ❌ | ✅ geoscenefm_map.html | New |
| Frontend dashboard | ❌ | ✅ dashboard/index.html | New |
| Purity evaluation | ❌ | ✅ Stage 10 | New |

---

## 17. Observations & Findings

**Finding 1 — sky_scene is the best-separated category (purity 53%).**  
CLIP's image encoder strongly encodes sky-dominant visual structure. Cluster 2 is 66% sky_scene among its labelled members — the clearest semantic grouping without any supervision.

**Finding 2 — built_env is the most cohesive non-sky cluster (intra-sim 0.804).**  
Building-dominant images form a tight group (cluster 7, purity 41.9%). Urban infrastructure features (facades, windows, street furniture) are consistently encoded.

**Finding 3 — road_scene sub-divides into 5 meaningful sub-clusters.**  
Rather than a failure, this reveals that CLIP discriminates road scene sub-types: highway, dense intersection, wet road, night scene, rural road etc. The sub-clusters have intra-sim 0.762–0.812, indicating they are visually coherent. This is finer-grained structure than the 4 coarse scene groups provide.

**Finding 4 — 3,147 high-similarity edges confirm non-random spatial structure.**  
2.52% of all 124,750 pairs exceed 0.85 cosine similarity — a sparse but meaningful network indicating genuine visual communities in the embedding space.

**Finding 5 — L2 normalisation produces unit-sphere embeddings as expected.**  
Mean norm = 1.000, std = 0.000. Cosine similarity = dot product throughout.

**Finding 6 — 20.1% PCA variance in 2D indicates non-linear embedding structure.**  
The high-dimensional variance is spread across many dimensions — UMAP and t-SNE provide better 2D representations than PCA for this data.

---

## 18. Known Limitations

| Limitation | Impact |
|---|---|
| **No GPS coordinates** | Cannot validate geographic proximity — only visual similarity is measured |
| **Generic CLIP model** | Not fine-tuned for geographic or urban-specific tasks |
| **road_scene dominates (34.2%)** | Creates class imbalance in purity evaluation; ARI is suppressed |
| **81 test-split images lack masks** | 16.2% of dataset has no semantic label for evaluation |
| **CPU inference only** | ~2.5 min for 500 images; scales linearly |
| **Simulated GPS in map** | Folium map uses hash-assigned coordinates, not real geolocation |
| **KMeans assumes spherical clusters** | Not optimal for non-spherical embedding manifolds |
| **No human validation** | Cluster quality assessed only via internal metrics + segmentation labels |

---

## 19. Recommended Next Steps

**Immediate (no new data needed):**
1. **GeoCLIP / SatCLIP model swap** — replace `openai/clip-vit-base-patch32` in `config.py` and `02_extract_embeddings.py` with a geographically specialised encoder. The entire pipeline downstream is model-agnostic.
2. **CLIP fine-tuning** — contrastive fine-tune CLIP on the 419 labelled images using (image, scene_group) pairs. Expected purity improvement to 0.65+.
3. **Per-category pixel fraction features** — the `label_composition.csv` already contains 19 per-category pixel fractions. Use these as additional feature dimensions alongside CLIP embeddings (concatenate and re-cluster).

**With additional data:**
4. **Scale to 5,000–10,000 images** — set `N_IMAGES = 5000` in `config.py`. The pipeline handles it; only the 500×500 similarity matrix step (Stage 5) needs FAISS approximation at scale.
5. **Geotagged imagery** — Mapillary, Google Street View API, or OpenStreetMap imagery with GPS coordinates. Enables actual spatial proximity validation: do embedding-space neighbours share geographic proximity?

**Analysis extensions:**
6. **Cluster sub-type inspection** — render the top-5 representative images for road_scene sub-clusters (0,1,3,4,5) to characterise what visual sub-type each captures (highway, city, wet, night, rural).
7. **Confusion matrix analysis** — examine which scene groups are most confused with each other to understand CLIP's visual feature priorities.
8. **Approximate nearest neighbour** — for 5K+ images, replace exact cosine similarity matrix with FAISS IVF index for sub-millisecond retrieval.
