# This code snippet demonstrates the concept of overfitting and underfitting in machine learning models. It simulates training and validation loss over 10 epochs, showing how the model's performance changes over time.
# Training loss keeps decreasing.
# Validation loss starts increasing.
# That moment indicates:
# Overfitting has begun.

import numpy as np

epochs = np.arange(1, 11)

training_loss = np.array([
    1.20, 0.90, 0.70, 0.55, 0.40,
    0.32, 0.25, 0.20, 0.16, 0.12
])

validation_loss = np.array([
    1.25, 0.95, 0.75, 0.60, 0.52,
    0.50, 0.55, 0.63, 0.74, 0.90
])

for epoch, train, val in zip(
    epochs,
    training_loss,
    validation_loss
):
    print(
        f"Epoch {epoch:2d} | "
        f"Train Loss: {train:.2f} | "
        f"Validation Loss: {val:.2f}"
    )

    """
    How Do We Prevent Overfitting?

We'll learn these one by one in the coming lessons:

Early Stopping
Dropout
L2 Regularization
Data Augmentation
More Data
Simpler Models

These are some of the most valuable techniques in practical deep learning.
    """