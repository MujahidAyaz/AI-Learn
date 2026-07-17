"""
============================================================
Training Engine
============================================================

This module contains the training and validation logic
for the Fashion-MNIST classifier.

Responsibilities
----------------
1. Train the model
2. Validate the model
3. Return loss and accuracy
============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch


# ==========================================================
# Training Function
# ==========================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """

    # Set model to training mode
    model.train()

    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:

        # Move tensors to CPU/GPU
        images = images.to(device)
        labels = labels.to(device)

        # Step 1
        optimizer.zero_grad()

        # Step 2
        outputs = model(images)

        # Step 3
        loss = criterion(outputs, labels)

        # Step 4
        loss.backward()

        # Step 5
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Get predicted class
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        correct_predictions += (predictions == labels).sum().item()

        total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (
        correct_predictions / total_samples
    ) * 100

    return epoch_loss, epoch_accuracy


# ==========================================================
# Validation Function
# ==========================================================

def validate(model, dataloader, criterion, device):
    """
    Evaluates the model.
    """

    model.eval()

    running_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (
        correct_predictions / total_samples
    ) * 100

    return epoch_loss, epoch_accuracy