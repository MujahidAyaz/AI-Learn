"""
============================================================
Visualization Module
============================================================

This module generates training visualizations
for the Fashion-MNIST project.

Responsibilities
----------------
1. Plot Loss Curve
2. Plot Accuracy Curve
3. Save figures automatically

Author : Mujahid Ayaz
Repository : AI-Learn
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

from pathlib import Path

import matplotlib.pyplot as plt

from configs.config import OUTPUT_DIR


# ==========================================================
# Plot Loss Curve
# ==========================================================

def plot_loss(history):
    """
    Plot training and validation loss.
    """

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    save_path = OUTPUT_DIR / "plots" / "loss_curve.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# Plot Accuracy Curve
# ==========================================================

def plot_accuracy(history):
    """
    Plot training and validation accuracy.
    """

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.legend()

    plt.grid(True)

    save_path = OUTPUT_DIR / "plots" / "accuracy_curve.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# Generate All Plots
# ==========================================================

def generate_training_plots(history):
    """
    Generate every training plot.
    """

    plot_loss(history)

    plot_accuracy(history)

    print("\nTraining plots saved successfully!")