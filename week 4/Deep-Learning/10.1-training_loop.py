# Complete Code

import numpy as np
np.random.seed(42)

# Dataset
x = np.array([1, 2, 3, 4], dtype=np.float32)
y_true = np.array([2, 4, 6, 8], dtype=np.float32)

# Random Weight
weight = np.random.randn()

# Hyperparameters
learning_rate = 0.01
epochs = 100

print(f"Initial Weight: {weight:.4f}")
print("-" *45)

for epoch in range(epochs):
    # Forward Pass
    y_pred = weight * x

    # Loss Calculation (Mean Squared Error)
    loss = np.mean((y_true - y_pred) ** 2)

    # Backpropagation
    gradient = (-2/len(x)) * np.sum(x * (y_true - y_pred))

    # Update Weight
    weight = weight - learning_rate * gradient

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}: Weight: {weight:.4f}, Loss: {loss:.4f}")
