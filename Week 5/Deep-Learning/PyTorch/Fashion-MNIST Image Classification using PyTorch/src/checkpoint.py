"""
============================================================
Checkpoint Module

This module contains helper functions for saving and loading
PyTorch model checkpoints.
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
from pathlib import Path

from configs.config import MODEL_DIR
from src.logger import logger


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(model, filename="best_model.pth"):
    """
    Saves only the model weights.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    filename : str
        Name of checkpoint file.
    """

    checkpoint_path = MODEL_DIR / filename

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    logger.info("Best model saved successfully.")

    print(checkpoint_path)