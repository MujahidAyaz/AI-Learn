"""
============================================================
Training History Utilities
============================================================

Save training metrics to a CSV file.
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import csv

from configs.config import OUTPUT_DIR
from src.logger import logger


# ==========================================================
# Save History
# ==========================================================

def save_history(history):
    """
    Save training history as CSV.
    """

    output_file = OUTPUT_DIR / "training_history.csv"

    with open(output_file, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ]
        )

        for epoch in range(len(history["train_loss"])):

            writer.writerow(
                [
                    epoch + 1,
                    history["train_loss"][epoch],
                    history["train_accuracy"][epoch],
                    history["val_loss"][epoch],
                    history["val_accuracy"][epoch],
                ]
            )

    logger.info(f"Training history saved to: {output_file}")