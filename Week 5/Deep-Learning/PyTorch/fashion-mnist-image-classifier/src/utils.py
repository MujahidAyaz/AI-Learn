"""
============================================================
Utility Functions
============================================================

This module contains reusable helper functions used across
the Fashion-MNIST project.

Responsibilities
----------------
1. Create project directories
2. Set random seed
3. Display dataset information
4. Visualize sample images

Author : Mujahid Ayaz
Repository : AI-Learn
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import random
import torch
import matplotlib.pyplot as plt

from configs.config import (
    DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    RANDOM_SEED,
)

# ==========================================================
# Create Project Directories
# ==========================================================

def create_directories():
    """
    Creates project directories if they do not already exist.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Set Random Seed
# ==========================================================

def set_random_seed():
    """
    Sets random seed for reproducibility.
    """

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================================
# Display Dataset Information
# ==========================================================

def display_dataset_info(train_dataset, test_dataset):

    print("=" * 60)
    print("Fashion-MNIST Dataset Summary")
    print("=" * 60)

    print(f"Training Samples : {len(train_dataset)}")
    print(f"Testing Samples  : {len(test_dataset)}")

    print(f"\nImage Shape      : {train_dataset[0][0].shape}")

    print("\nClasses")

    for index, class_name in enumerate(train_dataset.classes):
        print(f"{index} : {class_name}")


# ==========================================================
# Visualize Sample Images
# ==========================================================

def visualize_samples(dataset, num_images=9):
    """
    Displays sample images from the dataset.
    """

    plt.figure(figsize=(8, 8))

    for i in range(num_images):

        image, label = dataset[i]

        plt.subplot(3, 3, i + 1)

        plt.imshow(image.squeeze(), cmap="gray")

        plt.title(dataset.classes[label])

        plt.axis("off")

    plt.tight_layout()

    plt.show()