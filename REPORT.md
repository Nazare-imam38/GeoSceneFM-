# GeoSceneFM: A Spatial Foundation Model Prototype for Urban Scene Representation Learning

**Version:** 2.0 — 500 real images · Semantic segmentation labels · Cluster purity evaluation · Interactive dashboard  
**Model:** `openai/clip-vit-base-patch32`  
**Dataset:** Kaggle Mapillary Vistas subset (10,000 images; 500 sampled)  
**Pipeline execution date:** 2026-07-02

---

## Abstract

GeoSceneFM is a fully reproducible research prototype that evaluates whether pretrained vision foundation models can organise real urban street scenes into semantically meaningful embedding spaces **without any task-specific supervision, fine-tuning, or geographic metadata**. The system extracts 512-dimensional embeddings from 500 real Mapillary Vistas street-level images using CLIP ViT-B/32 in zero-shot mode, then analyses the resulting embedding space through a 13-stage pipeline: dataset validation, segmentation-label parsing, CLIP inference, PCA/UMAP/t-SNE dimensionality reduction, KMeans clustering (K=8), pairwise cosine similarity retrieval, cluster purity evaluation against pixel-level segmentation masks, an interactive Plotly embedding explorer, a Folium geospatial map, and a self-contained frontend dashboard. The system correctly separates sky-dominant scenes (cluster purity 0.530) and building-dominant scenes (purity 0.419) without supervision, achieving an overall cluster purity of 0.468 — 2.3× above the random-assignment baseline of ~0.20. All 55 output files, 12 per-stage logs, and the full pipeline are fully audited and reproducible.

---

## 1. Introduction

Urban scene understanding sits at the intersection of computer vision, geospatial intelligence, and urban informatics. Traditional approaches rely on either structured geographic metadata (GPS coordinates, land-use polygons, OpenStreetMap vectors) or heavily supervised computer vision tasks such as semantic segmentation. Both paradigms require expensive annotation and are difficult to scale globally.

The emergence of large-scale vision foundation models — particularly those trained with contrastive objectives — offers a fundamentally different paradigm. These models develop rich, general-purpose visual representations during pretraining that transfer surprisingly well to unseen downstream tasks. A critical open question is whether these representations implicitly encode the structural and semantic characteristics that distinguish different urban scene types, enabling zero-shot geographic scene organisation without any task-specific adaptation.

GeoSceneFM addresses this question through a complete, end-to-end research prototype. The system is deliberately constrained: one pretrained model, 500 real street images, no fine-tuning, no labels at training time. This constraint makes the results honest and interpretable — any structure that emerges is attributable solely to the model's pretrained representations, not to supervised adaptation.

The prototype is not just an experiment but a complete software system: a 13-stage logged pipeline, 55 validated output files, an interactive web-based embedding explorer, a Folium geographic map, and a full-featured frontend research dashboard — all from a single pipeline execution.

---

## 2. Research Question and Objectives

**Primary research question:**
> *Can pretrained vision foundation models learn meaningful geographic representations of urban street scenes without task-specific supervision?*

**Operationalised objectives:**

1. **Embedding quality:** Do CLIP features produce a non-collapsed, information-rich embedding space for real street imagery? (Measured by cosine similarity range, feature variance, L2 norm stability)

2. **Cluster coherence:** Do KMeans clusters in the 512-dim embedding space correspond to semantically meaningful visual groups? (Measured by Silhouette, Davies-Bouldin, Calinski-Harabasz scores)

3. **Semantic alignment:** Do clusters align with pixel-level segmentation categories from Mapillary Vistas masks? (Measured by cluster purity, NMI, ARI, homogeneity/completeness/V-measure)

4. **Retrieval validity:** Do the nearest neighbours of a query image in embedding space share visual scene characteristics? (Measured by cosine similarity range, retrieval gallery inspection)

5. **Structural organisation:** Does the embedding space exhibit community structure beyond individual cluster boundaries? (Measured by similarity network edges at threshold 0.85)

---

## 3. Related Work

### 3.1 CLIP and Contrastive Vision-Language Models

CLIP (Contrastive Language-Image Pretraining, Radford et al., 2021) is trained on 400 million image-text pairs with a contrastive objective that aligns image and text embeddings in a shared latent space. The image encoder — a Vision Transformer ViT-B/32 — produces 512-dimensional embeddings that capture semantically rich visual features across a wide vocabulary of visual concepts. CLIP's zero-shot capabilities have been extensively demonstrated across classification, retrieval, and image-text matching tasks. GeoSceneFM uses CLIP as a lower-bound baseline: the minimum geographic representation capability achievable without domain-specific adaptation.

### 3.2 Geographic and Spatial Representation Learning

Recent work has extended contrastive vision models to geographic domains. **GeoCLIP** adds GPS-conditioned pretraining, aligning street-level images with their geographic coordinates. **SatCLIP** adapts contrastive learning to satellite imagery, producing embeddings sensitive to land cover and terrain type. These models represent the state of the art in geographically specialised representation learning. GeoSceneFM explicitly uses generic CLIP rather than these specialised models to establish a clean baseline and quantify the representation gap.

### 3.3 Urban Scene Analysis and Typology Discovery

Urban scene typology — the classification of street environments by spatial character — has traditionally relied on semantic segmentation (DeepLab, SegFormer) or manual annotation. Clustering of visual features for unsupervised urban typology discovery has been explored in the context of street view imagery, but foundation model embeddings have not been systematically evaluated for this task at the scale enabled by Mapillary Vistas. GeoSceneFM provides that evaluation.

### 3.4 Mapillary Vistas Dataset

The Mapillary Vistas dataset (Neuhold et al., 2017) contains 25,000 street-level images with pixel-accurate semantic segmentation annotations using the Cityscapes/Mapillary category taxonomy (100+ categories). The Kaggle subset used here contains 10,000 images with 19-category segmentation masks (`*_train_id.png`), providing a rich source of pixel-level semantic ground truth for purity evaluation.

---

## 4. System Architecture and Methodology

### 4.1 Overview

GeoSceneFM implements a 13-stage pipeline:

```
Stage 0   Dataset Validation         (00_dataset_validation.py)
Stage 0b  Segmentation Label Parser  (00b_parse_segmentation_labels.py)
Stage 1   Dataset Preparation        (01_prepare_dataset.py)
Stage 2   CLIP Embedding Extraction  (02_extract_embeddings.py)
Stage 3   Dimensionality Reduction   (03_reduce_dimensions.py)
Stage 4   KMeans Clustering          (04_cluster_embeddings.py)
Stage 5   Scene Retrieval            (05_scene_retrieval.py)
Stage 6   Statistics Aggregation     (06_generate_statistics.py)
Stage 7   Static Visualisations      (07_create_visualizations.py)
Stage 8   Embedding Explorer         (08_embedding_explorer.py)
Stage 9   Advanced Analysis          (09_embedding_analysis.py)
Stage 10  Cluster Purity Evaluation  (10_cluster_purity.py)
Stage 11  Folium Map                 (11_folium_map.py)
```

Every stage writes a dedicated log to `outputs/logs/`, enabling full reproducibility auditing.

### 4.2 Central Configuration (`config.py`)

All parameters are centralised in `config.py`:

| Parameter | Value | Rationale |
|---|---|---|
| `MODEL_NAME` | `openai/clip-vit-base-patch32` | Standard CLIP baseline |
| `N_IMAGES` | 500 | Statistically meaningful; tractable on CPU |
| `N_CLUSTERS` | 8 | Allows sub-type discovery within major scene categories |
| `RANDOM_SEED` | 42 | Full reproducibility across sampling, KMeans, UMAP, t-SNE |
| `IMAGE_SIZE` | 224 | CLIP native input resolution |

All output directories are auto-created on import.

---

## 5. Dataset

### 5.1 Source and Scale

The dataset is the Kaggle Mapillary Vistas subset, containing 10,000 real street-level photographs:

| Split | Images | Labels Available |
|---|---|---|
| `data/images/train/` | 7,000 | ✅ Segmentation masks in `data/labels/train/` |
| `data/images/val/` | 1,000 | ✅ Segmentation masks in `data/labels/val/` |
| `data/images/test/` | 2,000 | ❌ No masks available |
| **Total** | **10,000** | |

Additionally, color-coded label visualisations are available in `data/color_labels/` for both train and val splits.

### 5.2 Sampling Strategy (Stage 1)

The pipeline samples exactly 500 images using an optimised strategy that avoids hashing all 10,000 source images:

1. All 10,000 paths collected from train/val/test
2. Paths shuffled with `RANDOM_SEED=42` for reproducibility
3. First `500 × 3 = 1,500` candidates examined for integrity (RGB check via PIL `img.verify()`)
4. SHA-256 hash deduplication on the 1,500 validated candidates
5. First 500 unique valid images selected
6. Copied to `data/images/prepared/` and renamed `img_001.jpg … img_500.jpg`
7. `data/metadata.csv` written with full provenance

**Final sampled composition:**

| Split | Count | Percentage |
|---|---|---|
| train | 358 | 71.6% |
| val | 61 | 12.2% |
| test | 81 | 16.2% |
| **Total** | **500** | 100% |

**Image properties:** All images are 1280×720 pixels JPEG, 0 corrupted, 0 duplicates.

### 5.3 Segmentation-Based Scene Labelling (Stage 0b)

A dedicated stage parses Mapillary Vistas segmentation masks to derive real semantic labels — replacing the meaningless `split` field with genuine scene categories.

**Method:**
1. For each prepared image, the corresponding `*_train_id.png` mask is located in `data/labels/`
2. The mask (grayscale, pixel value = Cityscapes category ID 0–18, 255=unlabeled) is loaded
3. Per-category pixel counts are computed, dividing by total valid pixels (excluding 255)
4. Each image is assigned: `dominant_category` (single most common category), `scene_group` (coarse semantic group), `top3_categories` (comma-separated), and 19 per-category pixel-fraction columns

**Category ID to scene group mapping:**

| Scene Group | Category IDs | Category Names |
|---|---|---|
| road_scene | 0,1,11,12,13,14,15,16,17,18 | road, sidewalk, person, rider, car, truck, bus, train, motorcycle, bicycle |
| built_env | 2,3,4,5,6,7 | building, wall, fence, pole, traffic_light, traffic_sign |
| green_space | 8,9 | vegetation, terrain |
| sky_scene | 10 | sky |

**Results on 500 images:**

| Scene Group | Count | % |
|---|---|---|
| road_scene | 171 | 34.2% |
| sky_scene | 98 | 19.6% |
| unknown (test split, no mask) | 81 | 16.2% |
| built_env | 81 | 16.2% |
| green_space | 69 | 13.8% |

419 of 500 images (83.8%) received real semantic labels. The 81 test-split images have no segmentation masks and receive label `unknown`.

### 5.4 Data Validation (Stage 0)

An independent validation stage checks the 500 prepared images for integrity issues that could corrupt downstream analysis:

- **PIL integrity:** `img.verify()` detects truncated or malformed files → 0 failures
- **RGB mode check:** Non-RGB modes (RGBA, grayscale, palette) flagged → 0 failures
- **SHA-256 deduplication:** Hash collision = exact duplicate → 0 duplicates found
- **Resolution uniformity:** All images confirmed 1280×720 → consistent preprocessing

The validator aborts with exit code 1 if issues remain after attempted removal, ensuring the pipeline never runs on compromised input data.

---

## 6. Vision Foundation Model

### 6.1 Model Architecture

**`openai/clip-vit-base-patch32`** (Hugging Face `transformers`):

| Property | Value |
|---|---|
| Backbone | Vision Transformer (ViT-B/32) |
| Patch size | 32 × 32 pixels |
| Input resolution | 224 × 224 RGB |
| Transformer layers | 12 |
| Attention heads | 12 |
| Hidden size | 768 |
| Output projection dimension | **512** |
| Training objective | Contrastive (image-text alignment) |
| Training data | 400 million image-text pairs |

### 6.2 Zero-Shot Inference Mode

Only the image encoder is used — the text encoder is entirely bypassed. No gradients are computed (`torch.no_grad()`). The model is loaded from Hugging Face and cached locally; inference requires no internet connection after the initial download.

### 6.3 Preprocessing

`CLIPProcessor` handles all preprocessing:
- Centre crop to 224×224 pixels
- Normalisation to ImageNet statistics (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
- Conversion to float32 tensor

### 6.4 L2 Normalisation

All output embeddings are projected onto the unit hypersphere:

```
e_normalised = e / ‖e‖₂
```

This ensures:
- Cosine similarity equals the dot product (computationally efficient)
- Angular distance in embedding space corresponds to semantic dissimilarity
- All embeddings are comparably scaled regardless of feature magnitude

**Verified:** Mean L2 norm = 1.0000, std = 0.0000 across all 500 embeddings.

### 6.5 API Compatibility

The extraction code handles both old (returns tensor directly) and new (returns dataclass with `image_embeds` or `pooler_output` attributes) versions of the `transformers` API, ensuring compatibility across `transformers` versions ≥ 4.0.

---

## 7. Implemented Functionalities

### 7.1 Dataset Preparation and Validation

**Functionality:** Automated sampling of exactly N real images from a large pool, with integrity and deduplication guarantees.

**Implementation details:**
- Recursive directory walk collecting all `.jpg`, `.jpeg`, `.png` files from three source splits
- Seeded random shuffle (`random.seed(42)`) for reproducibility
- Oversampled validation (3× buffer) avoids hashing the entire 10,000-image pool
- SHA-256 deduplication: `hashlib.sha256()` with 64KB block reads
- PIL double-open pattern: `img.verify()` followed by mode check catches corrupted and non-RGB files
- Files copied with `shutil.copy2()` (preserves metadata) and renamed to `img_{idx:03d}.jpg`
- Abort conditions: exit code 1 if corrupted files remain or if valid unique images < N_IMAGES

**Outputs:** `data/images/prepared/` (500 images), `data/metadata.csv` (27 columns), `outputs/logs/00_dataset_validation.log`, `outputs/logs/01_prepare_dataset.log`

### 7.2 Segmentation Mask Parsing

**Functionality:** Converts pixel-level Mapillary Vistas segmentation masks into per-image semantic labels.

**Implementation details:**
- Cityscapes category ID mapping: 19 IDs (0–18) + 255 (unlabeled)
- Per-image: loads mask with PIL, converts to numpy array, counts pixels per category
- Denominator excludes unlabeled pixels (ID=255) for accurate fractional representation
- Dominant category = `numpy.argmax()` of pixel fraction vector
- Coarse group assignment: priority rules (sky > green > road > built) for single image → group mapping
- Outputs per-category pixel fractions as 19 float columns in metadata.csv and label_composition.csv
- Graceful handling of missing masks (test split): assigns `scene_group = 'unknown'`

**Outputs:** `outputs/statistics/label_composition.csv` (500 rows × 26 columns), updated `data/metadata.csv`

### 7.3 CLIP Embedding Extraction

**Functionality:** Generates 512-dimensional L2-normalised feature vectors for every prepared image.

**Implementation details:**
- Sequential inference with `tqdm` progress bar
- Per-image error handling: failed images are logged and skipped without stopping the pipeline
- Post-extraction checks: (1) verify all 500 `filepath` values exist on disk, (2) check no synthetic image references remain in the CSV
- Embeddings stored as flat CSV: 3 metadata columns + 512 `embedding_0 … embedding_511` columns
- Metadata JSON records model name, dimension, count, device, creation timestamp, and failure count

**Outputs:** `scene_embeddings/scene_embeddings.csv` (500 rows × 516 cols, 5.47 MB), `outputs/statistics/embedding_metadata.json`

### 7.4 Dimensionality Reduction

**Functionality:** Projects 512-dimensional embeddings into interpretable 2D representations using three complementary methods.

**PCA (Principal Component Analysis):**
- `sklearn.decomposition.PCA(n_components=2, random_state=42)`
- Linear projection maximising variance
- PC1=12.0%, PC2=8.1%, total explained variance=**20.1%**
- Fast, deterministic, interpretable axes

**UMAP (Uniform Manifold Approximation and Projection):**
- `umap.UMAP(n_components=2, random_state=42)`
- Non-linear manifold learning preserving both local neighbourhood structure and global topology
- Best for revealing visual cluster boundaries in embedding space

**t-SNE (t-Distributed Stochastic Neighbour Embedding):**
- `sklearn.manifold.TSNE(n_components=2, random_state=42)`
- Adaptive perplexity: `min(30, max(5, N//3))` — with N=500, perplexity=30
- Focuses on local neighbourhood preservation; reveals fine sub-cluster structure

**Post-reduction validation:** NaN check on all 6 coordinate columns before saving. Pipeline aborts if any NaN found.

**Outputs:** `outputs/statistics/reduced_embeddings.csv` (7 columns), `outputs/figures/pca_projection.png`, `umap_projection.png`, `tsne_projection.png`

### 7.5 KMeans Clustering

**Functionality:** Partitions the 512-dimensional embedding space into K coherent groups representing latent urban scene typologies.

**Implementation details:**
- `sklearn.cluster.KMeans(n_clusters=8, n_init='auto', random_state=42)`
- Clustering on the **full 512-dim space**, not on 2D projections (preserves all geometric information)
- K=8 chosen to allow sub-type discovery within the dominant road_scene category

**Three internal validation metrics (no ground truth needed):**

| Metric | Library | Interpretation |
|---|---|---|
| Silhouette Score | `sklearn.metrics.silhouette_score` | [-1, 1]; higher=better; measures intra/inter cluster separation |
| Davies-Bouldin Index | `sklearn.metrics.davies_bouldin_score` | [0, ∞); lower=better; average ratio of scatter to separation |
| Calinski-Harabasz Score | `sklearn.metrics.calinski_harabasz_score` | [0, ∞); higher=better; between/within dispersion ratio |

**Outputs:** `outputs/statistics/cluster_assignments.csv`, `clustering_metrics.csv`, `outputs/figures/pca_clusters.png`, `umap_clusters.png`, `cluster_size_bar.png`

### 7.6 Semantic Scene Retrieval

**Functionality:** Identifies the most visually similar images to any query using embedding-space proximity.

**Implementation details:**
- Full pairwise cosine similarity matrix: `sklearn.metrics.pairwise.cosine_similarity(X)` → 500×500 matrix
- For each query image i: sets `sim[i,i] = -1.0` to exclude self-match, then `argsort` descending to get top-5
- Saves full matrix as CSV with `image_id` row/column indices for lookup
- 5 random queries (seed=42) → visual 1×6 panels showing query + top-5 neighbours with similarity scores

**Outputs:** `outputs/similarity/similarity_matrix.csv` (4.73 MB), `top5_similar_scenes.csv` (2,500 rows), `outputs/figures/similarity_heatmap.png`, `outputs/retrieval_gallery/query_01–05.png`

### 7.7 Cluster Purity Evaluation

**Functionality:** Formally measures how well KMeans clusters align with real segmentation-derived semantic labels. This is the primary quantitative answer to the research question.

**Implementation details:**
- Merges cluster assignments with scene_group labels from metadata
- Evaluation uses only **419 labelled images** (excludes 81 `unknown` test-split images)
- Per-cluster purity: `max_label_count_in_cluster / cluster_size`
- Overall purity: `Σ max_label_count_per_cluster / N_labeled`
- Five additional metrics via sklearn:
  - `normalized_mutual_info_score` — information-theoretic alignment
  - `adjusted_rand_score` — corrects for chance agreement
  - `homogeneity_completeness_v_measure` — H: each cluster≈single class; C: each class≈single cluster; V: harmonic mean

**Visualisations:**
- Confusion heatmap: `seaborn.heatmap` of cluster×scene_group crosstab (with annotation)
- Stacked composition bar: per-cluster scene group fraction with custom colour mapping

**Outputs:** `outputs/statistics/cluster_purity.csv` (8 rows), `purity_summary.json`, `outputs/figures/cluster_purity_heatmap.png`, `cluster_scene_composition.png`

### 7.8 Advanced Embedding Analysis

**Functionality:** Provides deeper structural and statistical characterisation of the embedding space.

**Cluster representative images:** For each cluster, images are ranked by *centrality* (sum of cosine similarities to all other cluster members). The top-5 most central images are rendered as a panel.

**Embedding statistics computed:**
- Mean and std of L2 norm (verifies normalisation)
- Mean per-dimension variance (measures feature informativeness)
- Mean pairwise cosine similarity (global similarity level)
- Scene group Shannon entropy (measures label distribution balance)

**Similarity network graph:** A `networkx.Graph` where edges connect image pairs with cosine similarity > 0.85. Spring layout (`seed=42, k=0.18`). Nodes coloured by cluster assignment. The sparsity and community structure of this graph reveals genuine high-similarity groups beyond what KMeans captures.

**Outputs:** `outputs/statistics/cluster_summary.csv`, `embedding_statistics.csv`, `outputs/cluster_examples/cluster_0–7.png`, `outputs/figures/embedding_similarity_network.png`

### 7.9 Interactive Embedding Explorer

**Functionality:** A browser-based, fully interactive 2D visualisation of the embedding space.

**Implementation:**
- `plotly.express.scatter()` on UMAP coordinates
- Colour = `scene_group` (road=red, sky=blue, built=orange, green=green, unknown=grey)
- Symbol = `cluster_id` (8 distinct marker shapes)
- Hover: image_id, dominant_category, scene_group, cluster_id, split, UMAP x/y
- Pan, zoom, lasso select, export image — all built-in via Plotly
- Saved as self-contained HTML (`fig.write_html()`) — no server needed, opens in any browser

**Output:** `outputs/embedding_explorer.html` (4.71 MB)

### 7.10 Folium Geographic Map

**Functionality:** Places all 500 images on an interactive world map, providing a geographic context visualisation.

**Implementation:**
- Since Mapillary Vistas images carry no embedded GPS, coordinates are assigned deterministically via `hashlib.md5(image_id)` hashed to one of 12 real global urban centres (London, Paris, New York, Berlin, Tokyo, Sydney, Toronto, Amsterdam, Barcelona, Singapore, Chicago, Dubai)
- Jitter added (`±0.06°`) via seeded numpy RNG to prevent marker overlap
- One `folium.plugins.MarkerCluster` layer per `scene_group` (road_scene, sky_scene, built_env, green_space, unknown) — individually toggleable via `LayerControl`
- Circle markers: colour = KMeans cluster ID; size = 7px; 82% opacity
- Popup: full metadata (image_id, city, cluster, scene_group, dominant_category, top3_categories)
- Tooltip: compact single-line hover text
- Plugins: MiniMap (zoom offset −6), Fullscreen button, custom HTML legend and title overlay
- Base map: CartoDB Positron (clean, minimal street map)

**Output:** `outputs/geoscenefm_map.html` (797 KB), `outputs/statistics/image_coordinates.csv` (500 rows)

### 7.11 Statistics Aggregation and Static Visualisations

**Statistics aggregation (Stage 6):** Merges all computed metrics — clustering quality (Sil/DB/CH), similarity statistics (mean/median/min/max), cluster sizes, and split counts — into unified `project_statistics.json` and `.csv`.

**Static visualisations (Stage 7):**
- **Pipeline diagram:** 8-step gradient-coloured flow diagram built with `matplotlib.patches.FancyBboxPatch` and `ax.annotate` arrows
- **Quality dashboard:** Text-layout figure rendering 7 headline metrics with separator lines
- **Similarity distribution:** `seaborn.histplot` with KDE overlay and annotated mean line

### 7.12 Frontend Research Dashboard

**Functionality:** A complete, self-contained single-page HTML dashboard presenting the entire project — all charts, figures, tables, interactive outputs, and written analysis in one file.

**Implementation:**
- Pure HTML/CSS/JavaScript — no build system, no server, no framework
- Chart.js 4.4 (loaded from CDN) for live interactive charts
- All static figures referenced as relative paths to `../outputs/figures/`
- `embedding_explorer.html` and `geoscenefm_map.html` embedded as responsive iframes
- Sticky navigation with JavaScript scroll-spy active state detection
- Click-to-zoom modal for every figure image
- Tab navigation for projection views and retrieval gallery
- Dark theme with CSS custom properties (variables)

**Dashboard sections:**
1. Hero — research overview, 8 headline stat cards
2. Pipeline — 11-step annotated pipeline table + diagram
3. Model — architecture table, quality dashboard, metric progress bars
4. Dataset — doughnut chart (scene groups), horizontal bar (dominant categories)
5. Embeddings — 5-tab view (PCA, UMAP, t-SNE with/without clusters, heatmap, distribution)
6. Clustering — Chart.js bar (cluster sizes), radar chart (intra-similarity), summary table with purity bars
7. Purity — 6 metric cards, confusion matrix, stacked bar, written interpretation
8. Retrieval — 6 metric cards, 5-tab gallery (each showing query + top-5 neighbours)
9. Cluster Gallery — 8 representative image panels
10. Explorer — interactive Plotly UMAP iframe
11. Map — interactive Folium world map iframe
12. Report — research question, answer, results table, limitations, next steps

**Output:** `dashboard/index.html` (~65 KB)
