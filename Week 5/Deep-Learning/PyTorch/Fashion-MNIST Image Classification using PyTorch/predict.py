"""
============================================================
Prediction Script
============================================================

Loads a trained Fashion-MNIST model and predicts
classes for test images.

Author : Mujahid Ayaz
Repository : AI-Learn
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
import matplotlib.pyplot as plt

from configs.config import DEVICE, MODEL_DIR, BATCH_SIZE
from src.dataset import create_dataloaders
from src.model import FashionClassifier


# ==========================================================
# Class Names
# ==========================================================

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ==========================================================
# Load Model
# ==========================================================

def load_model():
    """
    Load the trained model.
    """

    model = FashionClassifier()

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "best_model.pth",
            map_location=DEVICE,
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


# ==========================================================
# Predict Sample Images
# ==========================================================

def predict_samples(model, test_loader, num_images=5):
    """
    Display predictions for a few test images.
    """

    images, labels = next(iter(test_loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = torch.max(probabilities, dim=1)

    plt.figure(figsize=(15, 3))

    for i in range(num_images):

        plt.subplot(1, num_images, i + 1)

        plt.imshow(images[i].cpu().squeeze(), cmap="gray")

        plt.axis("off")

        plt.title(
            f"P: {CLASS_NAMES[predictions[i]]}\n"
            f"T: {CLASS_NAMES[labels[i]]}\n"
            f"{confidence[i].item()*100:.1f}%"
        )

    plt.tight_layout()

    plt.show()


# ==========================================================
# Main
# ==========================================================

def main():

    _, test_loader, _, _ = create_dataloaders()

    model = load_model()

    predict_samples(model, test_loader)


if __name__ == "__main__":
    main()