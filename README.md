# GeoSceneFM: A Spatial Foundation Model Prototype for Urban Scene Representation Learning

GeoSceneFM evaluates whether pretrained vision foundation models can organize real urban street scenes into semantically meaningful geographic embedding spaces — **without any task-specific supervision or fine-tuning**.

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Project Architecture](#2-project-architecture)
3. [Dataset](#3-dataset)
4. [Model](#4-model)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Detailed Implementation](#6-detailed-implementation)
7. [Configuration](#7-configuration)
8. [Outputs Reference](#8-outputs-reference)
9. [Experimental Results](#9-experimental-results)
10. [Installation & Running](#10-installation--running)
11. [Dependencies](#11-dependencies)
12. [Limitations](#12-limitations)
13. [Future Work](#13-future-work)

---

## 1. Research Question

> *Can pretrained vision foundation models learn meaningful geographic representations of urban street scenes without task-specific supervision?*

GeoSceneFM approaches this through pure **zero-shot representation learning** — the CLIP image encoder generates 512-dimensional embedding vectors from real Kaggle street imagery. These embeddings are analyzed through dimensionality reduction, unsupervised clustering, and cosine similarity retrieval.

---

## 2. Project Architecture

```
GeoSceneFM/
├── config.py                          # Central configuration & directory setup
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── REPORT.md                          # Academic-style research report
├── PROJECT_AUDIT.md                   # Full pipeline audit with real metrics
│
├── data/
│   ├── images/
│   │   ├── prepared/                  # 60 sampled working images (pipeline target)
│   │   ├── train/                     # 7,000 real Kaggle training images
│   │   ├── val/                       # 1,000 real Kaggle validation images
│   │   └── test/                      # 2,000 real Kaggle test images
│   ├── labels/
│   │   ├── train/                     # ~7,000 semantic segmentation masks
│   │   └── val/                       # ~1,000 semantic segmentation masks
│   ├── color_labels/
│   │   ├── train/                     # 7,000 color-coded label maps
│   │   └── val/                       # 1,000 color-coded label maps
│   ├── archive.zip                    # Original Kaggle archive
│   └── metadata.csv                   # Generated: image_id, filepath, split, etc.
│
├── scene_embeddings/
│   └── scene_embeddings.csv           # 60 rows × 515 cols (512-dim CLIP embeddings)
│
├── outputs/
│   ├── figures/                       # 11 generated PNG visualizations
│   ├── statistics/                    # 8 CSV/JSON metric files
│   ├── similarity/                    # Similarity matrix + top-5 neighbors
│   ├── logs/                          # Per-stage stdout/stderr logs (10 files)
│   ├── retrieval_gallery/             # 5 visual query-result panels
│   ├── cluster_examples/              # 4 representative image panels per cluster
│   └── embedding_explorer.html        # Interactive Plotly UMAP visualization
│
└── src/
    ├── 00_download_images.py          # Synthetic image generator (dev/testing only)
    ├── 00_dataset_validation.py       # Dataset integrity checks (RGB, SHA-256 dedup)
    ├── 01_prepare_dataset.py          # Sample 60 real images → data/images/prepared/
    ├── 02_extract_embeddings.py       # CLIP ViT-B/32 embedding extraction
    ├── 03_reduce_dimensions.py        # PCA / UMAP / t-SNE projections
    ├── 04_cluster_embeddings.py       # KMeans(K=4) + Silhouette/DB/CH metrics
    ├── 05_scene_retrieval.py          # Cosine similarity matrix + top-5 retrieval
    ├── 06_generate_statistics.py      # Unified statistics aggregation
    ├── 07_create_visualizations.py    # Pipeline diagram + dashboard + distribution
    ├── 08_embedding_explorer.py       # Interactive Plotly HTML (embedding_explorer.html)
    ├── 09_embedding_analysis.py       # Cluster summary + network graph + statistics
    └── run_all.py                     # Full pipeline orchestrator
```

---

## 3. Dataset

The pipeline uses **60 real images** sampled from the Kaggle Mapillary Vistas subset:

| Source Split | Count |
|---|---|
| train | 46 |
| val | 7 |
| test | 7 |
| **Total** | **60** |

All images are real street-level photographs at **1280×720 pixels** (JPEG). The images are randomly sampled (seed=42) from 10,000 available real images across three splits. Sampling validates each candidate for RGB integrity and SHA-256 uniqueness before selection.

The full Kaggle repository also contains:
- ~8,000 semantic segmentation masks in `data/labels/` (Mapillary Vistas format, `*_train_id.png`)
- ~8,000 color-coded label maps in `data/color_labels/` (`*_train_color.png`)

These provide pixel-level category annotations for future supervised evaluation.

> **Development note:** `00_download_images.py` generates synthetic color-noise placeholder images for pipeline testing when real imagery is unavailable. It should never be used for experimental analysis — results on synthetic images are trivially inflated and not representative of real-world performance.

---

## 4. Model

**OpenAI CLIP — `openai/clip-vit-base-patch32`** (Hugging Face `transformers`)

| Property | Value |
|---|---|
| Architecture | Vision Transformer (ViT-B/32) |
| Patch size | 32×32 pixels |
| Input resolution | 224×224 RGB (resized by CLIPProcessor) |
| Output dimension | **512** |
| Normalization | L2 — unit sphere projection |
| Usage mode | Zero-shot, image encoder only |
| Fine-tuning | None |

---

## 5. Pipeline Overview

```
Real Kaggle Street Images (10,000 available)
    │
    ▼
[01] Dataset Preparation
    │  → Sample 60 from 300 validated candidates (oversampled 5×)
    │  → SHA-256 dedup + RGB integrity check
    │  → Copy to data/images/prepared/, write metadata.csv
    │
    ▼
[00] Dataset Validation
    │  → Checks all 60 prepared images (RGB, integrity, SHA-256 dedup)
    │  → Removes duplicates, continues if ≥60 valid remain
    │  → Resolution confirmed: 1280×720 (all identical)
    │
    ▼
[02] CLIP Embedding Extraction
    │  → CLIPProcessor → 224×224 input tensor
    │  → CLIPModel.get_image_features() → 512-dim vector
    │  → L2 normalize → unit sphere
    │  → scene_embeddings.csv (60 × 515 columns)
    │
    ▼
[03] Dimensionality Reduction
    │  → PCA: 23.3% total variance (PC1=14.3%, PC2=9.0%)
    │  → UMAP: topology-preserving manifold projection
    │  → t-SNE: perplexity=20, local cluster focus
    │  → reduced_embeddings.csv + 3 scatter plots
    │
    ▼
[04] KMeans Clustering (K=4)
    │  → On full 512-dim space
    │  → Silhouette=0.049, DB=2.744, CH=4.49
    │  → Cluster sizes: 18, 8, 22, 12
    │  → cluster_assignments.csv + 3 figures
    │
    ▼
[05] Scene Retrieval
    │  → 60×60 cosine similarity matrix
    │  → Mean sim=0.709, max=0.896, min=0.457
    │  → Top-5 nearest neighbours per image (300 records)
    │  → Heatmap + 5-query retrieval gallery
    │
    ▼
[06] Statistics Aggregation
    │  → project_statistics.json/.csv
    │
    ▼
[07] Static Visualizations
    │  → Pipeline diagram, quality dashboard, similarity distribution
    │
    ▼
[08] Interactive Explorer
    │  → Plotly UMAP scatter → outputs/embedding_explorer.html
    │
    ▼
[09] Advanced Analysis
       → Cluster summary + intra-cluster similarity
       → Per-cluster representative image panels
       → Embedding statistics (norm, variance, entropy)
       → Similarity network: 60 nodes, 36 edges (sim > 0.85)
```

---

## 6. Detailed Implementation

### Stage 0: Dataset Validation (`00_dataset_validation.py`)

Validates `data/images/prepared/` after Stage 1 runs.

- **Integrity check:** PIL `img.verify()` + RGB mode check
- **Duplicate detection:** SHA-256 hash, removes duplicates and continues (does NOT abort on duplicates unless fewer than 60 valid remain after removal)
- **Abort condition:** Only if valid unique images < `N_IMAGES=60`
- **Result:** 60 images, 0 corrupted, 0 duplicates, all 1280×720

### Stage 1: Dataset Preparation (`01_prepare_dataset.py`)

Sources real Kaggle images efficiently without hashing all 10,000.

1. Collects all 10,000 paths from `data/images/train/`, `val/`, `test/`
2. Shuffles with `RANDOM_SEED=42`
3. Validates the first `N_IMAGES × 5 = 300` candidates (integrity + RGB)
4. SHA-256 deduplicates those 300 candidates
5. Takes first 60 unique valid candidates
6. Copies to `data/images/prepared/` renamed `img_001.jpg` … `img_060.jpg`
7. Writes `data/metadata.csv` with: `image_id`, `filename`, `filepath`, `source_path`, `split`, `source_dataset`, `scene_label`

### Stage 2: Embedding Extraction (`02_extract_embeddings.py`)

- Pre-validates all 60 filepaths exist on disk before running model
- Loads CLIP via `CLIPModel.from_pretrained()` + `CLIPProcessor.from_pretrained()`
- Handles both old (tensor) and new (dataclass) `transformers` API variants
- L2 normalizes: `features / features.norm(p=2, dim=-1, keepdim=True)`
- Post-run: verifies no synthetic image references remain in the CSV
- Saves `scene_embeddings/scene_embeddings.csv` (60 rows × 515 cols)

### Stage 3: Dimensionality Reduction (`03_reduce_dimensions.py`)

| Method | Key config | Outcome |
|---|---|---|
| PCA | `n_components=2` | PC1=14.3%, PC2=9.0% explained variance |
| UMAP | `n_components=2, random_state=42` | Topology-preserving 2D layout |
| t-SNE | `perplexity=min(30, max(5, N//3))=20` | Local-cluster 2D layout |

Post-run NaN check on `reduced_embeddings.csv` — confirmed NaN-free.

### Stage 4: Clustering (`04_cluster_embeddings.py`)

KMeans on the full 512-dim space (not 2D projections). Three internal validation metrics computed. Results with real Mapillary Vistas data:

| Metric | Value |
|---|---|
| Silhouette Score | 0.049 |
| Davies-Bouldin Index | 2.744 |
| Calinski-Harabasz Score | 4.49 |

Low scores reflect the genuine challenge: real street images share substantial visual structure (road surfaces, sky, infrastructure) in the absence of fine-tuning or named category labels.

### Stage 5: Scene Retrieval (`05_scene_retrieval.py`)

- Full 60×60 cosine similarity matrix via `sklearn.metrics.pairwise.cosine_similarity`
- Top-5 nearest neighbours per image (self excluded by setting `sim[i,i]=-1`)
- Real similarity statistics: mean=0.709, median=0.710, min=0.457, max=0.896
- 5-query retrieval gallery (1280×720 real images displayed at native resolution)

### Stages 6–9

Statistics aggregation, three static summary figures, interactive Plotly HTML explorer saved as `outputs/embedding_explorer.html`, and advanced analysis including a NetworkX similarity graph (60 nodes, 36 edges at threshold 0.85).

---

## 7. Configuration (`config.py`)

| Parameter | Value |
|---|---|
| `MODEL_NAME` | `openai/clip-vit-base-patch32` |
| `IMAGE_SIZE` | 224 (CLIP input size) |
| `N_IMAGES` | 60 |
| `N_CLUSTERS` | 4 |
| `RANDOM_SEED` | 42 |
| `KAGGLE_TRAIN_DIR` | `data/images/train/` |
| `KAGGLE_VAL_DIR` | `data/images/val/` |
| `KAGGLE_TEST_DIR` | `data/images/test/` |
| `IMAGE_DIR` | `data/images/prepared/` |

All output directories auto-created by `config.py`.

---

## 8. Outputs Reference

| File | Location | Description |
|---|---|---|
| `metadata.csv` | `data/` | 60 rows: image_id, filepath, split |
| `scene_embeddings.csv` | `scene_embeddings/` | 512-dim embeddings, 60 rows |
| `reduced_embeddings.csv` | `outputs/statistics/` | PCA/UMAP/t-SNE coords + cluster_id |
| `cluster_assignments.csv` | `outputs/statistics/` | image_id → cluster_id |
| `clustering_metrics.csv` | `outputs/statistics/` | Sil/DB/CH scores |
| `project_statistics.json/csv` | `outputs/statistics/` | All metrics unified |
| `embedding_statistics.csv` | `outputs/statistics/` | Norm, variance, entropy |
| `cluster_summary.csv` | `outputs/statistics/` | Per-cluster intra-similarity |
| `embedding_metadata.json` | `outputs/statistics/` | Model, dim, count, timestamp |
| `similarity_matrix.csv` | `outputs/similarity/` | 60×60 cosine similarity |
| `top5_similar_scenes.csv` | `outputs/similarity/` | Top-5 per image (300 rows) |
| `pca_projection.png` | `outputs/figures/` | PCA 2D scatter |
| `umap_projection.png` | `outputs/figures/` | UMAP 2D scatter |
| `tsne_projection.png` | `outputs/figures/` | t-SNE 2D scatter |
| `pca_clusters.png` | `outputs/figures/` | PCA colored by cluster |
| `umap_clusters.png` | `outputs/figures/` | UMAP colored by cluster |
| `cluster_size_bar.png` | `outputs/figures/` | Images per cluster bar chart |
| `similarity_heatmap.png` | `outputs/figures/` | 60×60 heatmap |
| `cosine_similarity_distribution.png` | `outputs/figures/` | Histogram + KDE |
| `embedding_pipeline_diagram.png` | `outputs/figures/` | 8-step flow diagram |
| `embedding_quality_dashboard.png` | `outputs/figures/` | Metrics dashboard |
| `embedding_similarity_network.png` | `outputs/figures/` | NetworkX graph |
| `query_01–05.png` | `outputs/retrieval_gallery/` | Retrieval panels |
| `cluster_0–3.png` | `outputs/cluster_examples/` | Representative images |
| `embedding_explorer.html` | `outputs/` | Interactive Plotly UMAP |

---

## 9. Experimental Results

Results from the full clean pipeline run on real Kaggle data (2026-07-02):

| Metric | Value |
|---|---|
| Images | 60 |
| Source | Kaggle Mapillary Vistas subset |
| Resolution | 1280×720 |
| Embedding dimension | 512 |
| PCA explained variance (2 PCs) | 23.3% |
| Mean pairwise cosine similarity | **0.7088** |
| Median pairwise cosine similarity | 0.7095 |
| Min pairwise cosine similarity | 0.4565 |
| Max pairwise cosine similarity | 0.8957 |
| Silhouette Score (K=4) | **0.0492** |
| Davies-Bouldin Index (K=4) | **2.744** |
| Calinski-Harabasz Score (K=4) | **4.49** |
| Similarity network edges (> 0.85) | 36 / 1,770 possible |
| Mean embedding L2 norm | 1.0 (normalized) |

**Interpretation:** Low clustering scores (Silhouette≈0.05) are an honest result for generic CLIP applied to unlabeled Mapillary Vistas street imagery — all images share similar visual structure (roads, sky, infrastructure). The 36 high-similarity edges (>0.85) confirm that genuine structural clusters do exist; they are just not strongly separated at K=4 without category labels or fine-tuning.

---

## 10. Installation & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (real Kaggle images must be present in data/images/)
python src/run_all.py

# Or step by step
python src/01_prepare_dataset.py
python src/00_dataset_validation.py
python src/02_extract_embeddings.py
python src/03_reduce_dimensions.py
python src/04_cluster_embeddings.py
python src/05_scene_retrieval.py
python src/06_generate_statistics.py
python src/07_create_visualizations.py
python src/08_embedding_explorer.py
python src/09_embedding_analysis.py
```

Open `outputs/embedding_explorer.html` in any browser for the interactive UMAP view.
Logs for every stage are saved to `outputs/logs/`.

---

## 11. Dependencies

| Package | Role |
|---|---|
| `torch`, `torchvision` | Deep learning backend |
| `transformers` | CLIP model (Hugging Face) |
| `pillow` | Image loading |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | PCA, t-SNE, KMeans, metrics |
| `umap-learn` | UMAP |
| `matplotlib`, `seaborn` | Static plots |
| `plotly` | Interactive HTML explorer |
| `networkx` | Similarity graph |
| `tqdm` | Progress bars |
| `scipy` | Shannon entropy |

---

## 12. Limitations

- **No named scene categories in source data:** Mapillary Vistas uses segmentation masks, not filename-level category labels. The `split` field (train/val/test) is the only available compositional metadata.
- **Generic CLIP features:** Not fine-tuned for urban/geographic tasks — captures general visual similarity rather than semantic scene type.
- **60-image sample:** Small dataset limits statistical robustness.
- **No GPS coordinates:** No geographic localization or proximity validation.
- **CPU inference only (in this run):** Embedding extraction is slower without CUDA.

---

## 13. Future Work

- Parse Mapillary Vistas segmentation masks to assign dominant scene labels per image
- Replace CLIP with GeoCLIP or SatCLIP for geography-specialised features
- Scale to 1,000+ geotagged images
- Add GPS coordinates for spatial proximity validation
- Fine-tune CLIP on labeled urban scene categories
- Integrate satellite imagery and OpenStreetMap vector data
#   G e o S c e n e F M -  
 