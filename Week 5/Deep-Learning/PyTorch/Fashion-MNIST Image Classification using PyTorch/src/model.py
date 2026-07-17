"""
============================================================
Model Definition
============================================================
This module defines the neural network architecture used
for classifying Fashion-MNIST images.
Architecture
------------
Input Image (1 × 28 × 28)
↓
Flatten
↓
Linear (784 → 128)
↓
ReLU
↓
Linear (128 → 64)
↓
ReLU
↓
Linear (64 → 10)
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch.nn as nn


# ==========================================================
# FashionMNIST Classifier
# ==========================================================

class FashionClassifier(nn.Module):
    """
    Fully Connected Neural Network
    for Fashion-MNIST Classification.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Convert 28x28 image into 784 features
            nn.Flatten(),

            # First Hidden Layer
            nn.Linear(28 * 28, 128),

            nn.ReLU(),

            # Second Hidden Layer
            nn.Linear(128, 64),

            nn.ReLU(),

            # Output Layer
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """
        Defines how data flows
        through the neural network.
        """

        return self.network(x)