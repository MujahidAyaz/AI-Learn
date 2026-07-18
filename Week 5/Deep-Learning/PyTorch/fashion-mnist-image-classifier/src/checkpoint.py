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

from configs.config import MODEL_DIR
from src.logger import logger


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(model, filename="best_model.pth"):
    """
    Save model weights.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    filename : str
        Checkpoint filename.
    """

    checkpoint_path = MODEL_DIR / filename

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    logger.info(
        f"Model checkpoint saved: {checkpoint_path}"
    )