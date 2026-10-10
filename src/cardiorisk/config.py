"""Central configuration: paths, column definitions and defaults.

Everything that more than one module needs lives here so that training,
serving and tests agree on the same feature contract.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("CARDIORISK_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV = PROCESSED_DIR / "heart_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# The four "processed" files published by UCI. Each one comes from a
# different hospital; we keep the origin as a column for analysis only.
SOURCES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}

# Column order of the 14-attribute UCI files.
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

TARGET = "target"           # 1 = heart disease present (num > 0)
SOURCE_COL = "source"       # hospital of origin, NOT used as a model feature

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Engineered feature names (created inside the sklearn pipeline).
ENGINEERED_FEATURES = ["hr_reserve_pct", "age_x_oldpeak"]

# Physiologically plausible ranges used for cleaning and API validation.
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (60, 250),
    "chol": (80, 650),
    "thalach": (50, 230),
    "oldpeak": (-3.0, 7.0),
}

CATEGORY_LEVELS = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}

RANDOM_STATE = int(os.getenv("CARDIORISK_SEED", "42"))
