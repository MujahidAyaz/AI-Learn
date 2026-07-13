"""
Parameters vs Hyperparameters

This distinction is fundamental.

Parameters

These are learned automatically during training.
Examples:
Weights (W)
Biases (b)
You never manually set these values.
Gradient descent learns them.

Hyperparameters
These are chosen before training starts.
Examples:

Learning Rate
Batch Size
Epochs
Optimizer
Hidden Layers
Hidden Units
Dropout Rate
Weight Initialization

The model does not learn these.
You decide them.
"""
learning_rates = [0.1, 0.01, 0.001]

batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for batch in batch_sizes:
        print(
            f"Training with LR={lr}, Batch Size={batch}"
        )
