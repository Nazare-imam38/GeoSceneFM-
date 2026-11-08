# Future Research Questions & Graduate Study Proposals: GeoSceneFM

**Candidate:** Nazar E Imam (B.E. Geoinformatics Engineering, NUST)  
**Target Degrees:** Fully Funded Master's of Science (M.S.) and Doctor of Philosophy (Ph.D.) in the United States  
**Specializations:** GIScience, Geospatial Artificial Intelligence (GeoAI), Spatial Foundation Models, Remote Sensing, Urban Analytics  
**Document Code:** `RES-QUESTIONS-GEOSCENEFM-2026-v2`  

---

## Central Research Question 1 (Master's Thesis Focus)

> **"How does cross-modal geographic contrastive alignment between street-level perspective imagery and nadir high-resolution satellite orthoimagery improve zero-shot urban land-use typology classification compared to unimodal vision foundation models?"**

### 1. Specific Scientific / Computational Problem
Unimodal foundation models like CLIP process street-level images with ground-level perspective biases (occlusions, horizon dominance, foreground vehicle clutter) while failing to capture macroscopic spatial layout, parcel boundaries, and building footprints visible only from overhead remote sensing platforms. Conversely, satellite foundation models (e.g., SatCLIP, Prithvi) lack human-scale street-level architectural texture and street furniture resolution.

### 2. Motivating Project Finding & Limitation
In GeoSceneFM, generic CLIP achieved an overall cluster purity of 0.468, but struggled with class overlap between suburban roads and rural green spaces because ground-level views lack aerial context. Furthermore, the absence of paired overhead imagery prevented the model from resolving parcel-level land-use typologies.

### 3. Why the Problem Matters
Autonomous multimodal land-use classification is essential for rapid urban master planning, disaster damage assessment, property taxation, and tracking informal urban expansions in rapidly growing metropolitan regions where field surveys are cost-prohibitive.

### 4. Proposed Scientific & Computational Approach
Develop a dual-stream cross-modal contrastive learning architecture (`GeoCross-CLIP`):
- **Stream A:** Vision Transformer processing ground-level street view images (e.g., Mapillary / Google Street View).
- **Stream B:** Swin Transformer or ViT processing paired $0.5\,\text{m}$ satellite orthoimagery (NAIP or Sentinel-2/WorldView).
- **Cross-Modal Objective:** Symmetric InfoNCE contrastive loss conditioned on geographic proximity:
  $$\mathcal{L}_{\text{cross}} = -\log \frac{\exp(\langle \mathbf{e}_{\text{street}}, \mathbf{e}_{\text{sat}} \rangle / \tau)}{\sum_j \exp(\langle \mathbf{e}_{\text{street}}, \mathbf{e}_{\text{sat}, j} \rangle / \tau)}$$

### 5. Benchmark Datasets
- **CVUSA (Cross-View USA)** or **CVACT:** Paired ground-level and aerial imagery.
- **NAIP (National Agriculture Imagery Program):** $0.6\,\text{m}$ aerial orthoimagery.
- **OpenStreetMap (OSM):** Vector land-use polygons as ground truth.

### 6. Testable Hypothesis
*Conditioning street-level embeddings on paired nadir aerial context will increase zero-shot urban scene cluster purity from $0.468$ to $>0.680$ and resolve the ambiguity between residential and mixed-use commercial corridors.*

### 7. Rigorous Evaluation Strategy
- Measure overall cluster purity, Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI) against OSM land-use ground truth across five distinct metropolitan regions.
- Evaluate cross-modal retrieval (street-to-aerial and aerial-to-street Recall@1, Recall@5).

### 8. Expected Scientific Contribution
A validated multimodal representation learning framework that bridges the perspective divide between Earth observation remote sensing and human-scale street view informatics.

### 9. Suitability for a Master's Thesis
Highly structured, computationally feasible on a single modern GPU workstation (e.g., NVIDIA RTX 4090 or A100) within a 12-to-18-month research timeline, utilizing accessible open-access cross-view benchmarks.

### 10. Potential Extension into Ph.D. Research
Can be extended into a multi-temporal foundation model that monitors 4D urban spatiotemporal change (e.g., tracking gentrification, post-disaster recovery, or urban canopy loss over multi-year intervals).

---

## Central Research Question 2 (Ph.D. Dissertation Focus)

> **"Can spatially explicit distance-decay contrastive loss functions eliminate geographic distribution shift and regional morphological bias in global geospatial foundation models without requiring localized fine-tuning?"**

### 1. Specific Scientific / Computational Problem
Existing vision foundation models suffer from severe **spatial distribution shift** (spatial non-stationarity). Models pretrained predominantly on North American and Western European streetscapes fail dramatically when deployed in dense, informal, or heterogeneously planned cities in South Asia, Latin America, or Sub-Saharan Africa. Conventional loss functions treat negative pairs uniformly, ignoring Tobler's First Law of Geography: *"everything is related to everything else, but near things are more related than distant things."*

### 2. Motivating Project Finding & Limitation
GeoSceneFM demonstrated that generic CLIP relies heavily on Western urban cues (paved curbs, standardized lane markings, distinct building facades in Cluster 7). When deployed globally (simulated in Stage 11 across 12 world cities), a generic encoder cannot account for cultural and regional variance in urban forms.

### 3. Why the Problem Matters
GeoAI tools used for global disaster resilience, climate vulnerability mapping, and United Nations Sustainable Development Goals (SDGs) risk severe algorithmic bias, systematically under-characterizing the world's most vulnerable informal urban settlements.

### 4. Proposed Scientific & Computational Approach
Formulate a **Spatially Conditioned Geographic InfoNCE Loss ($\text{S-InfoNCE}$)** that penalizes embedding distance as a continuous function of geodesic spatial distance ($d_{\text{geo}}$) on the Earth's ellipsoid:
$$\mathcal{L}_{\text{S-InfoNCE}} = -\log \frac{\exp(\langle \mathbf{e}_i, \mathbf{e}_j \rangle / \tau)}{\sum_k w(d_{\text{geo}}(i, k)) \cdot \exp(\langle \mathbf{e}_i, \mathbf{e}_k \rangle / \tau)}$$
where $w(d_{\text{geo}}) = 1 - \exp\left(-\frac{d_{\text{geo}}^2}{2\ell^2}\right)$ re-weights negative samples based on spatial spatial lag ($\ell$). Integrate spherical coordinate position encodings (Space2Vec / spherical harmonics) directly into transformer self-attention blocks.

### 5. Benchmark Datasets
- **Global Mapillary Street-Level Imagery:** $100,000+$ geotagged images sampled across six continents (North America, South America, Europe, Africa, South Asia, East Asia).
- **Global Human Settlement Layer (GHSL) & World Settlement Footprint (WSF):** Global reference layers for morphological density and degree of urbanization.

### 6. Testable Hypothesis
*Weighting contrastive negative sample penalties with a spatial distance-decay function will reduce cross-continental typology classification error by $>35\%$ compared to standard contrastive foundation models.*

### 7. Rigorous Evaluation Strategy
- Measure Out-of-Distribution (OOD) generalization by training on Global North cities and testing zero-shot transfer on Global South cities.
- Compute Spatial Autocorrelation (Moran's $I$) of embedding residual errors to verify that the model eliminates localized spatial error clustering.

### 8. Expected Scientific Contribution
A foundational mathematical formulation integrating spatial non-stationarity and distance-decay principles directly into the loss functions of deep vision transformers.

### 9. Suitability for a Master's Thesis
Portions (e.g., testing distance-decay loss on a two-city benchmark) are feasible for an ambitious M.S. student; however, the theoretical formulation and multi-continental dataset scaling are inherently doctoral-level.

### 10. Potential Extension into Ph.D. Research
Forms the core theoretical pillar of a Ph.D. dissertation in GeoAI and Spatial Computing, establishing the student as an authority on spatial bias mitigation in foundation models.

---

## Central Research Question 3 (Methodological & Applied GeoAI Focus)

> **"To what extent can self-supervised foundation model embeddings replace supervised semantic segmentation for automated urban flood vulnerability and surface impermeability index calculation?"**

### 1. Specific Scientific / Computational Problem
Calculating urban surface runoff and flood vulnerability traditionally requires high-cost supervised semantic segmentation to calculate the exact percentage of impervious surfaces (asphalt, concrete) versus pervious ground (soil, vegetation). These pixel-level segmentation models fail under adverse atmospheric conditions, standing floodwaters, or novel geographic regions.

### 2. Motivating Project Finding & Limitation
GeoSceneFM's Stage 0b successfully extracted 19 continuous category pixel fractions (including road, sidewalk, building, vegetation, terrain, and sky) from segmentation masks. Crucially, the 512-dim CLIP embeddings in Stage 4 and 9 clustered images with high intra-cluster similarity according to these exact physical characteristics.

### 3. Why the Problem Matters
Urban pluvial flash flooding is accelerating under climate change. Municipal disaster management authorities require near-instantaneous runoff potential estimates from millions of crowdsourced street photographs during hurricane and monsoon events without waiting for slow pixel-level segmentation inference.

### 4. Proposed Scientific & Computational Approach
Develop a lightweight, zero-shot regression probe trained directly on frozen foundation model embeddings to predict continuous hydrological runoff coefficients ($C \in [0.1, 0.95]$) and surface permeability fractions:
- Extract $L_2$-normalized representations $\mathbf{e} \in \mathbb{R}^{512}$.
- Train an uncertainty-aware Bayesian Multi-Layer Perceptron (B-MLP) or Gaussian Process regressor mapping $\mathbf{e} \mapsto (C_{\text{runoff}}, \sigma_{\text{uncertainty}})$.
- Validate predictions against physical hydrologic runoff models (e.g., EPA SWMM - Storm Water Management Model).

### 5. Benchmark Datasets
- **Cityscapes & Mapillary Vistas:** Datasets with complete paired segmentation ground truth.
- **USGS Urban Hydrology Datasets:** In-situ streamflow and pluvial inundation records paired with street-level imagery.

### 6. Testable Hypothesis
*A lightweight regression probe operating on frozen foundation model embeddings can predict urban impervious surface fractions within $5\%$ of the accuracy of fully supervised DeepLabV3+ segmentation networks while reducing computational inference runtime by $90\%$.*

### 7. Rigorous Evaluation Strategy
- Compute Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$) between predicted runoff coefficients and true segmentation-derived coefficients.
- Measure inference frames-per-second (FPS) and GPU memory consumption.

### 8. Expected Scientific Contribution
A computationally efficient, uncertainty-quantified methodology for real-time natural hazard and climate resilience analytics utilizing spatial foundation models.

### 9. Suitability for a Master's Thesis
Ideal Master's thesis topic: combines applied machine learning, practical GIS hydrology, and foundation model probing with clear, quantifiable metrics and high real-world publication potential.

### 10. Potential Extension into Ph.D. Research
Can be scaled into a real-time, edge-deployable cyber-physical system integrating municipal traffic cameras, satellite synthetic aperture radar (SAR) flood masks, and street-level GeoAI models for dynamic disaster response routing.

