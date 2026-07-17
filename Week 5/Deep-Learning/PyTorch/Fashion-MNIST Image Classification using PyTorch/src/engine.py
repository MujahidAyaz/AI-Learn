"""
============================================================
Training Engine
============================================================

This module contains the complete training pipeline for the
Fashion-MNIST classifier.

Responsibilities
----------------
1. Train model for one epoch
2. Validate model
3. Manage complete training loop
4. Save best model checkpoint
5. Return training history

============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch

from src.checkpoint import save_checkpoint


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.

    dataloader : DataLoader
        Training dataloader.

    criterion :
        Loss function.

    optimizer :
        Optimizer.

    device :
        CPU or CUDA device.

    Returns
    -------
    tuple
        (epoch_loss, epoch_accuracy)
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

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


# ==========================================================
# Validation
# ==========================================================

def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate the model.

    Returns
    -------
    tuple
        (validation_loss, validation_accuracy)
    """

    model.eval()

    running_loss = 0.0
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


# ==========================================================
# Complete Training Pipeline
# ==========================================================

def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs,
    model_name,
):
    """
    Complete training pipeline.

    Parameters
    ----------
    model : torch.nn.Module

    train_loader : DataLoader

    test_loader : DataLoader

    criterion :
        Loss function.

    optimizer :
        Optimizer.

    device :
        CPU / CUDA

    epochs : int

    model_name : str

    Returns
    -------
    dict
        Training history.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_accuracy = 0.0

    print("\nStarting Training...\n")

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                model=model,
                filename=model_name,
            )

    print("\nTraining Completed Successfully!")

    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

    return history