from pathlib import Path
import os


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

RESULTS_DIR = PROJ_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42

EMBEDDER_NAME_MAP = {
    "timesfm": "timesfm",
    "MOMENT-1-base": "moment",
    "vae": "vae",
    "ts2vec": "ts2vec",
}

