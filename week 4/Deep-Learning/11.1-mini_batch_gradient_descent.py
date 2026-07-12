# Complete Code

import numpy as np

np.random.seed(42)

# Dataset
x = np.array([1,2,3,4,5,6,7,8], dtype=np.float32)
y = np.array([2,4,6,8,10,12,14,16], dtype=np.float32)

# Initialize Weight
weight = np.random.randn()

# Hyperparameters
learning_rate = 0.01
epochs = 100
batch_size = 2

print(f"Initial Weight: {weight:.4f}")
print("-" * 50)

# Training Loop
for epoch in range(epochs):

    for i in range(0, len(x), batch_size):

        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # Forward Pass
        prediction = weight * x_batch

        # Loss
        loss = np.mean((y_batch - prediction) ** 2)

        # Gradient
        gradient = (-2 / len(x_batch)) * np.sum(
            x_batch * (y_batch - prediction)
        )

        # Weight Update
        weight -= learning_rate * gradient

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss:.6f} | "
            f"Weight: {weight:.4f}"
        )

print("-" * 50)
print(f"Final Weight: {weight:.4f}")