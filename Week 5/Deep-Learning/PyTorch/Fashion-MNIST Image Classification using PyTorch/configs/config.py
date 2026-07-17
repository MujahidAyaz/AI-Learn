"""
============================================================
Project Configuration
============================================================

This module stores all configurable settings used throughout
the Fashion-MNIST project.

Keeping configuration in one place makes the project easier
to maintain and avoids hardcoded values across files.
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = PROJECT_ROOT / "data"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Saved model directory
MODEL_DIR = PROJECT_ROOT / "saved_models"

# ==========================================================
# Dataset Configuration
# ==========================================================

IMAGE_SIZE = (28, 28)

NUM_CHANNELS = 1

NUM_CLASSES = 10

CLASS_NAMES = [
    "T-Shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

# ==========================================================
# DataLoader Configuration
# ==========================================================

BATCH_SIZE = 64

SHUFFLE_TRAIN = True

SHUFFLE_TEST = False

NUM_WORKERS = 0

# ==========================================================
# Training Configuration
# ==========================================================

EPOCHS = 10

LEARNING_RATE = 0.001

# ==========================================================
# Device Configuration
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# Checkpoint Configuration
# ==========================================================

BEST_MODEL_NAME = "best_model.pth"

SAVE_BEST_ONLY = True

PRINT_EVERY = 1