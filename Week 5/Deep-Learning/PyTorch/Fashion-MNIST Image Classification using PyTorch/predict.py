"""
============================================================
Prediction Script
============================================================

Loads the trained Fashion-MNIST model and predicts
random test images.

Author : Mujahid Ayaz
============================================================
"""

# ==========================================================
# Imports
# ==========================================================

import random
import torch
import matplotlib.pyplot as plt

from configs.config import (
    DEVICE,
    MODEL_DIR,
    BEST_MODEL_NAME,
    OUTPUT_DIR,
    NUM_PREDICTIONS,
    PREDICTION_FIGURE,
)

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
# Prediction Visualization
# ==========================================================

def predict_samples(model, dataset):

    plt.figure(figsize=(20,10))

    for index in range(NUM_PREDICTIONS):

        random_index = random.randint(
            0,
            len(dataset)-1
        )

        image, label = dataset[random_index]

        input_tensor = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted = prediction.item()

        actual = label

        is_correct = predicted == actual

        plt.subplot(
            2,
            5,
            index+1
        )

        plt.imshow(
            image.squeeze(),
            cmap="gray"
        )

        plt.axis("off")

        if is_correct:

            color = "green"

            status = " Correct"

        else:

            color = "red"

            status = " Wrong"

        plt.title(
            f"{status}\n"
            f"Prediction : {CLASS_NAMES[predicted]}\n"
            f"Actual    :{CLASS_NAMES[actual]}\n"
            f"{confidence.item()*100:.2f}%",
            fontsize=9,
            color=color,
        )

    plt.tight_layout()

    save_path = (
        OUTPUT_DIR
        / "predictions"
        / PREDICTION_FIGURE
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ==========================================================
# Main
# ==========================================================

def main():

    _, test_loader, _, test_dataset = create_dataloaders()

    model = load_model()

    predict_samples(
        model,
        test_dataset,
    )

    evaluate_model(
        model=model,
        dataloader=test_loader,
        device=DEVICE,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    main()