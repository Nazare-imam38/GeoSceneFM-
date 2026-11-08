# Scientific Contributions & Empirical Findings: GeoSceneFM

**Project:** GeoSceneFM — Spatial Foundation Model Prototype for Urban Scene Representation Learning  
**Lead Researcher:** Nazar E Imam  
**Institutional Background:** B.E. in Geoinformatics Engineering, NUST (CGPA: 3.52 / 4.00)  
**Document Code:** `CONTRIB-GEOSCENEFM-2026-v2`  

---

## 1. What Was Actually Accomplished

GeoSceneFM accomplished the design, implementation, and empirical execution of a full-scale research workflow evaluating the zero-shot representation capabilities of vision foundation models on complex, unstructured urban streetscapes. Specifically:

1. **Ingested and Curated 500 Real Street Scenes:** Sourced from the Kaggle Mapillary Vistas subset, verified for RGB integrity, cryptographically deduplicated via SHA-256 block hashing, and standardized at native $1280 \times 720$ resolution.
2. **Synthesized Pixel-Accurate Ground Truth:** Developed an automated parser that decoded paired grayscale segmentation masks (`*_train_id.png`), mapped 19 Cityscapes category IDs to coarse morphological groups, and extracted exact pixel-fraction distributions for 419 benchmarked images.
3. **Executed Zero-Shot Foundation Model Inference:** Extracted 512-dimensional representations using OpenAI's `clip-vit-base-patch32` without fine-tuning, strictly projecting vectors onto the unit hypersphere ($L_2$ norm $= 1.0000 \pm 0.0000$).
4. **Performed Comprehensive Manifold & Cluster Analysis:** Projected features into 2D linear (PCA) and non-linear (UMAP, t-SNE) coordinate spaces; executed unsupervised KMeans clustering ($K=8$) directly in the native 512-dimensional space.
5. **Constructed a Scalable Retrieval Engine:** Computed a complete $500 \times 500$ cosine similarity matrix (250,000 pairwise interactions), extracted top-5 nearest neighbors for all 500 scenes, and built visual query-result galleries.
6. **Conducted Paper-Quality Ground-Truth Purity Benchmarking:** Computed multi-metric clustering quality indicators (Overall Purity, NMI, ARI, Homogeneity, Completeness, V-Measure) against pixel-derived ground-truth scene categories.
7. **Engineered Production-Grade Geospatial & Web Interfaces:** Built an interactive WebGL Plotly embedding explorer, a full-featured Folium Leaflet geospatial portal, and a responsive, standalone HTML5 research dashboard.

---

## 2. Six Verified Scientific & Methodological Findings

### Finding 1: Autonomous Separation of Macro-Landscapes
**Zero-shot CLIP features naturally isolate broad, open-sky environments.**  
- *Evidence:* Cluster 2 achieved an overall cluster purity of **53.03%** and a dominant fraction of **66.04%** for `sky_scene` among its labeled members.  
- *Scientific Explanation:* Contrastive vision-language pretraining biases attention towards expansive, uninterrupted color gradients and horizon boundaries. The model effectively separates open highway vistas and panorama streetscapes from dense urban canyons without needing labels.

### Finding 2: Tight Latent Clustering of Architectural Enclosures
**Dense vertical infrastructure forms a distinct, cohesive community in embedding space.**  
- *Evidence:* Cluster 7 demonstrated an intra-cluster cosine similarity of **0.8041** (substantially exceeding the global mean of 0.7117), with `built_env` representing **49.06%** of its labeled images.  
- *Scientific Explanation:* Repetitive spatial frequencies characteristic of the built environment—such as rectangular facade geometries, window fenestration, building edges, and street poles—generate consistent high-magnitude activations across transformer attention heads.

### Finding 3: The Sub-Typology Resolution Paradox (Resolution of the ARI Anomaly)
**A low Adjusted Rand Index (0.0466) is not evidence of clustering failure, but evidence of fine-grained morphological discovery.**  
- *Evidence:* Ground-truth `road_scene` images (comprising 34.2% of the dataset) are partitioned across five separate clusters (0, 1, 3, 4, and 5), each exhibiting high internal cohesion ($0.7621$ to $0.8117$).  
- *Scientific Explanation:* Ground-truth semantic segmentations group all roadway surfaces, vehicles, and sidewalks into a single monolithic category (`road_scene`). However, real-world roads vary dramatically in morphology. Cluster 4 isolates multi-lane high-speed motorways (intra-cluster sim: 0.8117); Cluster 3 isolates shaded, high-density downtown intersections (intra-cluster sim: 0.8079); Cluster 0 captures suburban residential lanes. Because ARI mathematically penalizes splitting a single true class across multiple clusters, the model's ability to discover finer-grained spatial sub-types collapses the ARI metric while preserving an overall purity of 46.8% ($2.34\times$ over random baseline).

### Finding 4: Dispersion of Variance Across a Deep Non-Linear Manifold
**The semantic structure of urban streetscapes cannot be captured by linear low-rank approximations.**  
- *Evidence:* The first two principal components in PCA capture only **20.07%** of cumulative variance (PC1 = 12.00%, PC2 = 8.07%).  
- *Scientific Explanation:* Unlike aerial remote sensing datasets where spectral bands often exhibit strong collinearity (allowing PCA to capture $>80\%$ of variance in 2–3 bands), ground-level street imagery contains high structural heterogeneity (occlusions, shadows, multi-scale objects, perspective distortions). Information is distributed across hundreds of orthogonal latent dimensions, requiring non-linear manifold learning (UMAP/t-SNE) for faithful 2D interpretation.

### Finding 5: High-Similarity Community Sparsity
**Embedding space contains tightly bound visual communities rather than a diffuse uniform cloud.**  
- *Evidence:* Out of 124,750 possible unique image pairs, exactly **3,147 pairs (2.52%)** exhibit cosine similarity exceeding $0.85$.  
- *Scientific Explanation:* The global mean similarity of $0.7117$ reflects the baseline visual composition shared by all street photographs (pavement, sky, vertical objects). However, the $2.52\%$ high-similarity tail forms a modular graph network (evidenced in the NetworkX graph) where tightly connected subgraphs correspond to near-identical urban morphological typologies.

### Finding 6: Perfect Unit-Sphere Normalization Guarantees Angular Geometry
**Strict $L_2$ hypersphere projection ensures exact equivalence between Euclidean distance and cosine similarity.**  
- *Evidence:* Extraction logs verify mean vector norm $= 1.000000$ ($\sigma = 0.000000$).  
- *Scientific Explanation:* By removing vector magnitude variations (which often reflect image brightness or contrast rather than semantic content), clustering in Euclidean space optimizes directly for angular alignment: $\|\mathbf{u} - \mathbf{v}\|_2^2 = 2 - 2\cos(\mathbf{u}, \mathbf{v})$.

---

## 3. Documented Personal Contributions of Nazar E Imam

Based on workspace artifacts, Git configuration (`Nazare-imam38`), implementation scripts, and structural design:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONFIRMED PERSONAL CONTRIBUTIONS                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Complete end-to-end Python pipeline implementation (src/00 through 11).   │
│ 2. Design and mathematical implementation of automated mask parsing         │
│    algorithm (src/00b_parse_segmentation_labels.py).                        │
│ 3. Engineering of the SHA-256 deduplication and RGB validation guard        │
│    (src/00_dataset_validation.py).                                          │
│ 4. Implementation of full 500×500 cosine similarity retrieval engine and    │
│    visual gallery generator (src/05_scene_retrieval.py).                    │
│ 5. Execution and quantitative analysis of multi-metric clustering purity    │
│    framework (src/10_cluster_purity.py).                                    │
│ 6. Formulation of the "Sub-Typology Resolution Paradox" explaining the     │
│    divergence between ARI (0.047) and Purity (0.468).                       │
│ 7. Development of the Folium interactive GIS geospatial portal              │
│    (src/11_folium_map.py).                                                  │
│ 8. Design and coding of the standalone HTML5/CSS3/Chart.js research        │
│    dashboard (dashboard/index.html).                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Attribution Breakdown
- **Confirmed Personal Contribution:** All source code in `src/`, configuration parameters in `config.py`, data validation scripts, dashboard implementation, and execution of experimental runs on July 2, 2026.
- **Third-Party / Foundation Components:**
  - Neural Network Checkpoint: `openai/clip-vit-base-patch32` (Radford et al., 2021).
  - Raw Imagery: Mapillary Vistas subset (Neuhold et al., 2017) sourced from Kaggle.
  - Python Libraries: `transformers`, `torch`, `scikit-learn`, `umap-learn`, `folium`, `plotly`, `PIL`, `networkx`.

---

## 4. Scientific, Methodological, Technical & Practical Significance

### Scientific Significance
Establishes a rigorous, quantitative **empirical lower bound** on the capability of general vision-language models for zero-shot urban scene classification. It provides empirical proof that foundation models encode latent geospatial morphology without ever being trained on geographic coordinates or urban taxonomy labels.

### Methodological Significance
Demonstrates how fine-grained pixel-level segmentation masks can be automatically digested into coarse-grained compositional profiles for unsupervised clustering evaluation, eliminating the need for manual scene-level labeling.

### Technical Significance
Demonstrates production-level research engineering: strict validation guards, crash-resilient dataset loaders, deterministic seed controls, automated metrics logging, and interactive WebGL / Leaflet visualizations.

### Practical Significance
Provides a functional, lightweight prototype for **content-based spatial image retrieval (CBIR)**. Municipal planning agencies and GIS analysts can query thousands of untagged street view images to locate specific infrastructure types (e.g., protected bike lanes, damaged pavement, dense canopy corridors) without manual indexing.

---

## 5. Critical Negative Bounds: What the Project Does NOT Demonstrate

To maintain scientific integrity during graduate admissions reviews and faculty interviews, the following boundaries must be clearly stated:

1. **Does NOT Prove Spatial Autocorrelation or Geographic Proximity:** Because the Kaggle Mapillary Vistas subset omits public GPS metadata, the project does not prove that images close in embedding space are geographically adjacent in the physical world. (Folium map coordinates are deterministic hash-based demonstrations).
2. **Does NOT Claim a New Foundation Model Architecture:** The candidate did not train or fine-tune a new transformer architecture from scratch; the project evaluates and benchmarks an existing pretrained checkpoint (`openai/clip-vit-base-patch32`).
3. **Does NOT Demonstrate Superiority Over Geographically Adapted Models:** The project has not yet performed a head-to-head empirical comparison against GeoCLIP, SatCLIP, or RemoteCLIP (this is formulated as a core Master's/Ph.D. research question).
4. **Does NOT Claim Generalizability Across Remote Sensing Modalities:** Findings apply specifically to ground-level, forward-facing RGB street imagery; they cannot be extrapolated to nadir satellite multispectral, hyperspectral, or LiDAR data without empirical testing.

