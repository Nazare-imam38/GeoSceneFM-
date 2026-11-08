# Master Project File Manifest & Artifact Provenance

**Project:** GeoSceneFM — Urban Scene Representation Learning  
**Lead Researcher:** Nazar E Imam (B.E. Geoinformatics Engineering, NUST)  
**Workspace:** `c:\Projects\GeoScence\GeoSceneFM`  
**Execution Environment:** Windows 10 / 11, Python 3.13, PyTorch 2.x, Hugging Face `transformers`  
**Document Code:** `MANIFEST-GEOSCENEFM-2026-v2`  

---

## 1. Core Source Code Pipeline (`src/`)

All execution logic is modularized into sequentially indexed Python scripts located in `src/`.

| File Path | Lines of Code | Key Role & Scientific Significance |
|---|:---:|---|
| `src/00_download_images.py` | 55 | Developer utility for synthetic color-noise image generation. Documented as strictly restricted from experimental benchmark runs. |
| `src/00_dataset_validation.py` | 145 | Automated validation guard: verifies PIL integrity (`img.verify()`), RGB color mode, and SHA-256 deduplication across prepared images. Enforces abort if valid unique count $< N$. |
| `src/00b_parse_segmentation_labels.py` | 253 | Decodes paired Mapillary Vistas grayscale masks (`*_train_id.png`), mapping 19 Cityscapes category IDs to coarse scene groups and calculating per-category pixel fractions. |
| `src/01_prepare_dataset.py` | 205 | Optimized dataset sampler: shuffles 10,000 source images (`RANDOM_SEED=42`), inspects $3\times$ oversampled buffer (1,500 candidates), deduplicates, and extracts 500 images to `data/images/prepared/`. |
| `src/02_extract_embeddings.py` | 152 | Deep learning inference engine: loads `openai/clip-vit-base-patch32`, extracts 512-dim features in zero-shot mode, applies strict $L_2$ unit-sphere normalization, and writes `scene_embeddings.csv`. |
| `src/03_reduce_dimensions.py` | 121 | Manifold projection engine: computes linear PCA (PC1=12.0%, PC2=8.1%), non-linear UMAP ($n=2$), and adaptive t-SNE (perplexity=30); verifies NaN-free output coordinates. |
| `src/04_cluster_embeddings.py` | 156 | High-dimensional clustering engine: executes KMeans ($K=8$) directly on native 512-dim space; computes Silhouette (0.0564), Davies-Bouldin (3.107), and Calinski-Harabasz (23.043) metrics. |
| `src/05_scene_retrieval.py` | 142 | Information retrieval engine: constructs $500 \times 500$ cosine similarity matrix, ranks top-5 nearest neighbors for all 500 scenes (2,500 tuples), and generates visual query galleries. |
| `src/06_generate_statistics.py` | 85 | Unified data aggregation script: merges clustering metrics, similarity distributions, and dataset splits into consolidated `project_statistics.json` and `.csv`. |
| `src/07_create_visualizations.py` | 135 | Visualization generator: creates publication-grade pipeline flow diagram (`embedding_pipeline_diagram.png`), metrics dashboard, and KDE similarity distribution plots. |
| `src/08_embedding_explorer.py` | 78 | Interactive WebGL generator: renders Plotly Express 2D UMAP scatter colored by `scene_group` and symbolized by `cluster_id`, exporting to self-contained `embedding_explorer.html`. |
| `src/09_embedding_analysis.py` | 217 | Advanced structural analysis: computes cluster centrality, renders top-5 representative image panels (`cluster_0.png`–`cluster_7.png`), and generates a NetworkX similarity graph ($>0.85$ threshold). |
| `src/10_cluster_purity.py` | 216 | Quantitative ground-truth evaluation: calculates overall purity (0.4678), NMI (0.1014), ARI (0.0466), Homogeneity, Completeness, confusion matrix heatmap, and stacked composition bars. |
| `src/11_folium_map.py` | 254 | Interactive GIS portal generator: maps 500 scenes onto 12 world cities using deterministic MD5 hash mapping with spatial jitter ($\pm 0.06^\circ$); renders MarkerCluster layers to `geoscenefm_map.html`. |
| `src/run_all.py` | 77 | Master pipeline orchestrator: sequentially triggers pipeline stages using subprocess execution, checking exit status and piping output logs. |

---

## 2. Project Configuration & Dependencies

| File Path | Size | Description & Significance |
|---|:---:|---|
| `config.py` | 1.4 KB | Centralized global configuration: defines directory paths, model checkpoint (`openai/clip-vit-base-patch32`), sample size ($N=500$), cluster count ($K=8$), and random seed ($42$). |
| `requirements.txt` | 131 B | Python environment requirements: `torch`, `torchvision`, `transformers`, `pillow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `umap-learn`, `tqdm`, `folium`, `seaborn`, `plotly`, `networkx`, `requests`. |

---

## 3. Data Ingestion & Metadata Assets (`data/`)

| File Path | Size | Description & Scientific Role |
|---|:---:|---|
| `data/archive.zip` | 1.27 GB | Original Kaggle Mapillary Vistas compressed archive containing 10,000 street images and paired segmentation masks. |
| `data/metadata.csv` | 121 KB | Master dataset provenance catalog (500 rows × 27 columns): records `image_id`, source path, split, `dominant_category`, `scene_group`, `top3_categories`, and 19 per-category pixel fractions. |
| `data/images/prepared/` | ~120 MB | Working directory containing the 500 active experimental images (`img_001.jpg` … `img_500.jpg`), all verified at $1280 \times 720$ RGB JPEG. |
| `data/images/train/` | ~1.5 GB | Read-only source pool containing 7,000 real Kaggle training images. |
| `data/images/val/` | ~220 MB | Read-only source pool containing 1,000 real Kaggle validation images. |
| `data/images/test/` | ~440 MB | Read-only source pool containing 2,000 real Kaggle test images (unlabeled). |
| `data/labels/train/` | ~350 MB | Paired grayscale segmentation masks (`*_train_id.png`, pixel values 0–18). |
| `data/labels/val/` | ~50 MB | Paired grayscale validation segmentation masks. |

---

## 4. Extracted Representations & Statistics (`outputs/statistics/` & `scene_embeddings/`)

| File Path | Rows / Size | Analytical Content |
|---|:---:|---|
| `scene_embeddings/scene_embeddings.csv` | 500 rows × 516 cols (5.47 MB) | Primary representation matrix: stores `image_id`, filename, filepath, split, and 512 floating-point $L_2$-normalized CLIP embedding dimensions. |
| `outputs/statistics/embedding_metadata.json` | 15 lines | Extraction provenance: model name, dimension (512), image count (500), failure count (0), timestamp, and device (CPU). |
| `outputs/statistics/reduced_embeddings.csv` | 500 rows × 7 cols | 2D coordinate projections: `image_id`, `pca_x`, `pca_y`, `umap_x`, `umap_y`, `tsne_x`, `tsne_y`, and `cluster_id`. |
| `outputs/statistics/cluster_assignments.csv` | 500 rows × 2 cols | Discrete partition mapping: `image_id` to KMeans `cluster_id` ($0 \dots 7$). |
| `outputs/statistics/clustering_metrics.csv` | 1 row × 11 cols | Unsupervised clustering scores: Silhouette (0.0564), Davies-Bouldin (3.107), Calinski-Harabasz (23.043), and per-cluster sample counts. |
| `outputs/statistics/project_statistics.json` | 23 lines | Unified metrics JSON: consolidates image counts, splits, similarity min/max/mean/median, and clustering metrics. |
| `outputs/statistics/cluster_summary.csv` | 8 rows × 4 cols | Per-cluster profile: Cluster ID, Dominant Scene Group, Image Count, and Mean Intra-Cluster Cosine Similarity ($0.748$ to $0.812$). |
| `outputs/statistics/embedding_statistics.csv` | 1 row × 6 cols | Representation verification: Mean $L_2$ norm ($1.000$), std ($0.000$), mean feature variance ($0.000562$), and scene entropy ($1.549$ bits). |
| `outputs/statistics/label_composition.csv` | 500 rows × 26 cols | Full per-image segmentation breakdown: pixel fraction values across all 19 Cityscapes categories. |
| `outputs/statistics/cluster_purity.csv` | 8 rows × 11 cols | Quantitative purity table: Cluster ID, Size, Labeled Count, Dominant Scene, Dominant Fraction, Purity, and per-class fractions. |
| `outputs/statistics/purity_summary.json` | 12 lines | Headline validation metrics: Overall Purity (0.4678), NMI (0.1014), ARI (0.0466), Homogeneity (0.1299), Completeness (0.0832), V-Measure (0.1014). |
| `outputs/statistics/image_coordinates.csv` | 500 rows × 6 cols | Geocoded table: `image_id`, `city`, `country`, `lat`, `lon`, and `cluster_id`. |

---

## 5. Similarity & Information Retrieval Outputs (`outputs/similarity/`)

| File Path | Size | Description |
|---|:---:|---|
| `outputs/similarity/similarity_matrix.csv` | 4.73 MB | Full $500 \times 500$ pairwise cosine similarity matrix with labeled row and column image IDs. |
| `outputs/similarity/top5_similar_scenes.csv` | 425 KB | Ranked nearest-neighbor database (2,500 rows): query ID, matched ID, rank (1–5), cosine score, and filepaths. |

---

## 6. Execution Verification Logs (`outputs/logs/`)

| File Path | Size | Key Logged Execution Details |
|---|:---:|---|
| `outputs/logs/00_dataset_validation.log` | 1.3 KB | Stage 0 execution: 500 images scanned, 0 corrupted, 0 duplicates, resolution $1280 \times 720$ confirmed. |
| `outputs/logs/00b_parse_segmentation_labels.log` | 9.8 KB | Stage 0b execution: 419 masks resolved, 81 test images labeled unknown, category distribution tallied. |
| `outputs/logs/01_prepare_dataset.log` | 1.8 KB | Stage 1 execution: 1,500 candidates examined, SHA-256 deduplicated, 500 sampled and renamed. |
| `outputs/logs/02_extract_embeddings.log` | 6.5 KB | Stage 2 execution: CLIP ViT-B/32 inference on CPU, 500 extracted, zero synthetic references verified. |
| `outputs/logs/03_reduce_dimensions.log` | 1.5 KB | Stage 3 execution: PCA explained variance logged (PC1=12.0%, PC2=8.1%), UMAP and t-SNE completed without NaNs. |
| `outputs/logs/04_cluster_embeddings.log` | 2.5 KB | Stage 4 execution: KMeans ($K=8$) convergence, Sil/DB/CH metrics computed. |
| `outputs/logs/05_scene_retrieval.log` | 1.8 KB | Stage 5 execution: 500×500 matrix computed, top-5 neighbors extracted, 5 retrieval galleries saved. |
| `outputs/logs/06_generate_statistics.log` | 2.1 KB | Stage 6 execution: JSON and CSV consolidation verified. |
| `outputs/logs/07_create_visualizations.log` | 847 B | Stage 7 execution: Pipeline schematic, quality dashboard, similarity distribution saved. |
| `outputs/logs/08_embedding_explorer.log` | 622 B | Stage 8 execution: Plotly Express HTML rendering verified. |
| `outputs/logs/09_embedding_analysis.log` | 3.2 KB | Stage 9 execution: Centrality ranking, 8 cluster panels rendered, 3,147 graph edges extracted. |
| `outputs/logs/10_cluster_purity.log` | 3.1 KB | Stage 10 execution: Purity (0.4678), NMI (0.1014), ARI (0.0466) calculated on 419 labeled images. |
| `outputs/logs/11_folium_map.log` | 1.5 KB | Stage 11 execution: Hash assignment to 12 cities, spatial jittering, Folium HTML map written. |

---

## 7. Interactive Applications & Dashboards

| File Path | Technology | Size | Purpose & Features |
|---|---|:---:|---|
| `dashboard/index.html` | HTML5, CSS3, JavaScript, Chart.js 4.4 | ~65 KB | Standalone client-side research dashboard: 12 modular sections, interactive stat cards, live Chart.js charts, modal zoom, iframe viewers. |
| `outputs/embedding_explorer.html` | Plotly Express, WebGL | 4.71 MB | Interactive 2D UMAP scatter explorer: color-coded by scene group, symbolized by cluster ID, hover metadata. |
| `outputs/geoscenefm_map.html` | Folium, Leaflet.js | 797 KB | Geospatial interactive map: MarkerCluster layers per scene group, cluster color-coding, custom popups, MiniMap. |

---

## 8. Publication Figures (`outputs/figures/`)

| File Name | File Size | Description & Research Role |
|---|:---:|---|
| `cluster_purity_heatmap.png` | 182 KB | Confusion matrix heatmap cross-tabulating 8 KMeans clusters against ground-truth scene categories. |
| `cluster_scene_composition.png` | 135 KB | Stacked bar chart showing percentage composition of scene groups per cluster. |
| `cluster_size_bar.png` | 69 KB | Bar chart showing distribution of sample counts across all 8 clusters. |
| `cosine_similarity_distribution.png` | 141 KB | Histogram with Kernel Density Estimation (KDE) curve of all 124,750 pairwise similarities. |
| `embedding_pipeline_diagram.png` | 206 KB | Publication schematic illustrating the 8-stage core processing workflow. |
| `embedding_quality_dashboard.png` | 148 KB | Consolidated text-layout metric summary dashboard. |
| `embedding_similarity_network.png` | 1.50 MB | NetworkX spring layout graph displaying 3,147 high-similarity edges ($>0.85$ threshold). |
| `pca_clusters.png` | 609 KB | 2D scatter plot of first two principal components colored by cluster ID. |
| `pca_projection.png` | 320 KB | Raw PCA scatter plot showing linear variance distribution. |
| `similarity_heatmap.png` | 1.08 MB | High-resolution viridis heatmap of full $500 \times 500$ cosine similarity matrix. |
| `tsne_projection.png` | 331 KB | 2D t-SNE scatter plot (perplexity=30) demonstrating micro-neighborhood cluster preservation. |
| `umap_clusters.png` | 612 KB | 2D UMAP scatter plot colored by cluster ID demonstrating manifold separation. |
| `umap_projection.png` | 364 KB | Raw UMAP 2D manifold projection. |

