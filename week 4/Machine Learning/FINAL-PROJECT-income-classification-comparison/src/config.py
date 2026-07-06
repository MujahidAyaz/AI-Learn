"""
config.py
=========

Centralized configuration for the Adult Income Classification project.

All configurable values used throughout the project should be defined here.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

LOGS_DIR = PROJECT_ROOT / "logs"

# ==========================================================
# Dataset Configuration
# ==========================================================

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "adult/adult.data"
)

RAW_DATA_FILENAME = "adult_income.csv"

FILE_ENCODING = "utf-8"

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET_COLUMN = "income"

MISSING_VALUE_SYMBOL = "?"

TARGET_MAPPING = {
    "<=50K": 0,
    "<=50K.": 0,
    ">50K": 1,
    ">50K.": 1,
}

# ==========================================================
# Data Split Configuration
# ==========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42
SEED = RANDOM_STATE

# ==========================================================
# Cross Validation
# ==========================================================

CV_FOLDS = 5
N_JOBS = -1
GRID_SEARCH_VERBOSE = 1

# ==========================================================
# Model Hyperparameters
# ==========================================================

LOGISTIC_MAX_ITER = 1000

RF_ESTIMATORS = 200

XGB_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.10
XGB_MAX_DEPTH = 6

# ==========================================================
# Evaluation Metrics
# ==========================================================

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

# ==========================================================
# Visualization
# ==========================================================

FIGURE_DPI = 300

# ==========================================================
# Model Persistence
# ==========================================================

MODEL_EXTENSION = ".joblib"

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"
LOG_FILE_NAME = "project.log"

# ==========================================================
# Create Required Directories
# ==========================================================

for directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)