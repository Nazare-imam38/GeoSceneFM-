import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")

# Real Kaggle source directories (read-only sources)
KAGGLE_TRAIN_DIR = os.path.join(DATA_DIR, "images", "train")
KAGGLE_VAL_DIR   = os.path.join(DATA_DIR, "images", "val")
KAGGLE_TEST_DIR  = os.path.join(DATA_DIR, "images", "test")

# Prepared working directory (60 sampled images land here)
IMAGE_DIR = os.path.join(DATA_DIR, "images", "prepared")

# Label directories (for future scene-category annotation)
LABEL_DIR       = os.path.join(DATA_DIR, "labels")
COLOR_LABEL_DIR = os.path.join(DATA_DIR, "color_labels")

# Output directories
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR     = os.path.join(OUTPUT_DIR, "figures")
STATISTICS_DIR = os.path.join(OUTPUT_DIR, "statistics")
SIMILARITY_DIR = os.path.join(OUTPUT_DIR, "similarity")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")

# Scene Embedding directory
EMBEDDING_DIR = os.path.join(BASE_DIR, "scene_embeddings")

# Model configuration
MODEL_NAME  = "openai/clip-vit-base-patch32"
IMAGE_SIZE  = 224
N_IMAGES    = 500      # scaled up from 60
N_CLUSTERS  = 8        # more clusters for larger, labelled dataset
RANDOM_SEED = 42

# Ensure directories exist
for _d in [
    DATA_DIR, IMAGE_DIR, OUTPUT_DIR,
    FIGURE_DIR, STATISTICS_DIR, SIMILARITY_DIR,
    LOG_DIR, EMBEDDING_DIR,
]:
    os.makedirs(_d, exist_ok=True)
