"""
============================================================
Prediction Script
============================================================

Loads a trained Fashion-MNIST model and predicts
classes for test images.
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
import matplotlib.pyplot as plt

from configs.config import DEVICE, MODEL_DIR, BEST_MODEL_NAME
from src.dataset import create_dataloaders
from src.model import FashionClassifier
from src.metrics import evaluate_model

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
            MODEL_DIR / BEST_MODEL_NAME,
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
    
    evaluate_model(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    main()