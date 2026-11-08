# Research Auditor Clarification Questions for Project Owner

**To:** Nazar E Imam  
**From:** Senior Research Auditor & Academic Supervisor  
**Project:** GeoSceneFM — Urban Scene Representation Learning  
**Date:** October 2026  
**Document Code:** `QUESTIONS-OWNER-GEOSCENEFM-2026-v2`  

---

## Purpose of this Document
While the GeoSceneFM codebase is rigorously implemented, fully audited, and mathematically substantiated, university admissions committees and prospective faculty supervisors scrutinize student ownership, collaborative dynamics, and publication status. 

To ensure complete scientific defensibility during admissions interviews and written statements of purpose, please clarify the following nine questions:

---

### Category 1: Authorship, Independence & Collaborative Dynamics

#### Q1: Individual Development vs. Lab Collaboration
- **Audit Observation:** The Git configuration (`user.name=Nazare-imam38`) and all 13 Python scripts demonstrate a single, cohesive coding style and an individual commit trail on July 2, 2026.
- **Clarification Needed:** Was this project executed entirely as an independent research investigation, as part of an undergraduate course at NUST (e.g., Senior Design Project / FYP or Big Data Analytics), or in collaboration with a faculty advisor or peer research assistants?
- **Admissions Impact:** If independent, it directly demonstrates self-directed research maturity. If co-developed, collaborators and faculty advisors should be acknowledged accurately to avoid misattribution.

#### Q2: Current Formal Publication & Submission Status
- **Audit Observation:** The repository includes publication-grade reports (`REPORT.md`, `PROJECT_AUDIT.md`), but lacks formal preprint links (e.g., arXiv) or conference submission metadata.
- **Clarification Needed:** Has any manuscript based on GeoSceneFM been submitted, accepted, or presented at any workshop, conference, or journal (e.g., ACM SIGSPATIAL, IEEE IGARSS, AAG, or a regional GeoAI symposium)? Or is it currently an unpublished academic research prototype?
- **Admissions Impact:** If submitted or under review, include the formal citation and review status in your academic CV and statements of purpose.

---

### Category 2: Methodological Choices & Technical Clarifications

#### Q3: Choice of $K=8$ for KMeans Clustering
- **Audit Observation:** The pipeline scaled from $K=4$ (in the v1 60-image run) to $K=8$ (in the v2 500-image run).
- **Clarification Needed:** Was $K=8$ selected based on an empirical parameter sweep (such as an elbow curve across inertia or a silhouette optimization grid across $K \in [2, 16]$), or was it an analytical heuristic chosen to allow sub-clustering of the dominant `road_scene` category?
- **Admissions Impact:** Faculty interviewers frequently ask: *"How did you determine the optimal number of clusters?"* Having the rationale documented ensures a robust response.

#### Q4: Handling of the 81 Test-Split Images
- **Audit Observation:** Exactly 81 images originated from Kaggle's test split, which lacked segmentation masks and were assigned `unknown` (and excluded from the purity evaluation denominator).
- **Clarification Needed:** Did you consider running a pretrained segmentation network (e.g., SegFormer-B0 or DeepLabV3+ trained on Cityscapes) to pseudo-label these 81 images, or was your deliberate decision to preserve strict ground-truth purity by strictly using Mapillary's human-annotated masks?
- **Admissions Impact:** Defending the choice to avoid pseudo-labeling demonstrates a commitment to ground-truth integrity over artificial sample inflation.

#### Q5: Rationale for Selecting Generic CLIP over GeoCLIP or SatCLIP
- **Audit Observation:** The audit confirms that generic `openai/clip-vit-base-patch32` was used throughout.
- **Clarification Needed:** Did you conduct any preliminary experiments with geographically fine-tuned models like GeoCLIP (which uses GPS coordinates during pretraining), or was the focus strictly on establishing an unadapted baseline?
- **Admissions Impact:** Framing generic CLIP as an intentional "lower-bound empirical benchmark" transforms what could seem like a limitation into a deliberate, defensible methodological strategy.

---

### Category 3: Hardware, Deployment & Reproducibility Constraints

#### Q6: Computational Execution Environment & CUDA Availability
- **Audit Observation:** Log `02_extract_embeddings.log` records `Device: cpu` and an extraction runtime of ~2.5 minutes for 500 images.
- **Clarification Needed:** Was CPU execution chosen for portability across development environments, or did you encounter GPU memory constraints? If given access to a university GPU cluster, what scale ($N = 10,000$, $50,000$, etc.) would you target?
- **Admissions Impact:** Helps articulate your computational scaling roadmap when applying to high-performance computing (HPC) research laboratories.

#### Q7: Geographic Coordinate Strategy for Future Extensions
- **Audit Observation:** Stage 11 deterministic hash-based city mapping was a simulation for web GIS demonstration.
- **Clarification Needed:** Do you have access to a verified geotagged street view dataset (e.g., via the Mapillary API v4 or OpenStreetCam) for which real latitude/longitude coordinates can be ingested to evaluate spatial autocorrelation (Moran's $I$) in embedding space?
- **Admissions Impact:** Demonstrating readiness to ingest real geotagged data immediately elevates your proposal for Master's/Ph.D. research.

---

### Category 4: Academic Transcript & Coursework Alignment

#### Q8: Connection to NUST Undergraduate Coursework
- **Audit Observation:** You hold a B.E. in Geoinformatics Engineering from NUST with a CGPA of 3.52/4.00.
- **Clarification Needed:** Which specific upper-level courses at NUST (e.g., *Digital Image Processing*, *Photogrammetry & Remote Sensing*, *Spatial Databases*, *Machine Learning / Data Science*, *Spatial Data Analysis*) provided the conceptual foundation for this project?
- **Admissions Impact:** Allows us to explicitly connect your university transcript grades to your research outputs in your Statement of Purpose.

#### Q9: Long-Term Career Objective: Industry R&D vs. Academia
- **Audit Observation:** You work professionally on geospatial applications and interactive GIS systems, while preparing for fully funded U.S. graduate study.
- **Clarification Needed:** Is your ultimate career ambition to pursue a tenure-track academic professorship / research scientist role in GeoAI, or to lead advanced R&D teams in geospatial industry enterprises (e.g., Google Earth Engine, Esri Research, Planet Labs, Maxar)?
- **Admissions Impact:** Customizes the concluding paragraphs of your statements of purpose for Master's vs. Ph.D. committees.

