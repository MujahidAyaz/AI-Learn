"""
The Complete Training Pipeline
Every deep learning model, from a simple classifier to GPT, follows this loop:

Initialize Weights
        │
        ▼
Forward Propagation
        │
        ▼
Prediction
        │
        ▼
Loss Function
        │
        ▼
Backpropagation
        │
        ▼
Gradients
        │
        ▼
Gradient Descent
        │
        ▼
Updated Weights
        │
        ▼
Repeat...

Training is simply repeating this loop many times.
"""


# Step 1 — Import NumPy
import numpy as np
np.random.seed(42)

# Step 2 — Dataset
x = np.array([1, 2, 3, 4], dtype=np.float32)
y_true = np.array([2, 4, 6, 8], dtype=np.float32)

# Step 3 — Random Weight
weight = np.random.randn()
print("Initial Weight:", weight)

"""
Suppose it prints:  0.4967

The model is wrong.
Now it has to learn.
"""

# Step 4 — Hyperparameters
learning_rate = 0.01
epochs = 100

"""
Notice the new word:
Epoch

An epoch means:
One complete pass through the entire training dataset.
If we have 4 training examples and train for 100 epochs, the model sees those 4 examples 100 times.
"""
# Step 5 — The Training Loop
for epoch in range(epochs):

    # Everything happens inside this loop.
    # Step 6 — Forward Pass

    # Prediction:
    y_pred = weight * x
    
    #Step 7 — Loss
    # We'll use MSE.
    loss = np.mean((y_true - y_pred) ** 2)

    # Step 8 — Gradient
    # For this simple linear model, the gradient is:
    gradient = (-2 / len(x)) * np.sum(x * (y_true - y_pred))

    # Step 9 — Update Weight
    weight = weight - learning_rate * gradient
    # This is the learning step.

    # Step 10 — Print Progress
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Loss = {loss:.6f} | "
            f"Weight = {weight:.4f}"
        )

    # You should see the loss decreasing over time.
