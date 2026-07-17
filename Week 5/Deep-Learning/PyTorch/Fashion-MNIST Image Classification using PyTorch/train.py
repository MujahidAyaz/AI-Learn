"""
============================================================
Fashion-MNIST Image Classification using PyTorch
============================================================

Project Entry Point

Responsibilities
----------------
1. Create project directories
2. Set random seed
3. Load Fashion-MNIST dataset
4. Build neural network
5. Define loss function
6. Define optimizer
7. Train model
8. Save best model

============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from src.visualization import generate_training_plots
from tqdm import tqdm
from src.logger import logger

# ==========================================================
# Import Project Modules
# ==========================================================

from configs.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    BEST_MODEL_NAME,
)

from src.dataset import create_dataloaders
from src.model import FashionClassifier
from src.engine import train
from src.utils import (
    create_directories,
    set_random_seed,
    display_dataset_info,
    visualize_samples,
)


# ==========================================================
# Main Function
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("Fashion-MNIST Image Classification")
    logger.info("=" * 60)

    # ------------------------------------------------------
    # Project Setup
    # ------------------------------------------------------

    create_directories()

    set_random_seed()

    logger.info(f"Using Device : {DEVICE}")

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
    ) = create_dataloaders()

    display_dataset_info(
        train_dataset,
        test_dataset,
    )

    visualize_samples(train_dataset)

    # ------------------------------------------------------
    # Create Model
    # ------------------------------------------------------

    model = FashionClassifier().to(DEVICE)

    print("\nModel Architecture\n")
    print(model)

    # ------------------------------------------------------
    # Loss Function
    # ------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        model_name=BEST_MODEL_NAME,
    )


    generate_training_plots(history)

    logger.info("Training Finished Successfully!")

    return history


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    history = main()