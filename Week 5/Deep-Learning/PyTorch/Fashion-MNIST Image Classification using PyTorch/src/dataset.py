"""
============================================================
Dataset Module
============================================================

This module is responsible for:

1. Downloading Fashion-MNIST
2. Loading datasets
3. Creating DataLoaders
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from configs.config import (
    DATA_DIR,
    BATCH_SIZE,
    SHUFFLE_TRAIN,
    SHUFFLE_TEST,
    NUM_WORKERS,
)

# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.ToTensor()


# ==========================================================
# Dataset Loader Function
# ==========================================================

def load_dataset():
    """
    Downloads Fashion-MNIST (if required)
    and returns training and testing datasets.
    """

    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


# ==========================================================
# DataLoader Function
# ==========================================================

def create_dataloaders():
    """
    Creates PyTorch DataLoaders for
    training and testing.
    """

    train_dataset, test_dataset = load_dataset()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TEST,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
    )