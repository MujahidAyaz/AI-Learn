"""
============================================================
Evaluation Metrics
============================================================

This module evaluates the trained Fashion-MNIST model.

Responsibilities
----------------
1. Calculate Accuracy
2. Generate Classification Report
3. Plot Confusion Matrix
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from configs.config import OUTPUT_DIR


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate the trained model on the test dataset.
    """

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nTest Accuracy : {accuracy * 100:.2f}%")

    # ------------------------------------------------------
    # Classification Report
    # ------------------------------------------------------

    print("\nClassification Report\n")

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
    )

    print(report)

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    plt.figure(figsize=(10, 10))

    disp.plot(
        cmap="Blues",
        xticks_rotation=45,
    )

    plt.title("Confusion Matrix")

    save_path = (
        OUTPUT_DIR
        / "plots"
        / "confusion_matrix.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nConfusion matrix saved successfully!")