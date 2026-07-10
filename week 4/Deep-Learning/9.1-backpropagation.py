# Complete code for backpropagation algorithm

import numpy as np

# Training Sample
x = 3
y_true = 10

# Initial Weight
weight = 2.0

# Learning Rate
learning_rate = 0.1

# Forward Pass
prediction = weight * x

loss = (y_true - prediction) ** 2

print("Prediction:", prediction)
print("Loss:", loss)

# Backpropagation (Manual Gradient)
gradient = -2 * x * (y_true - prediction)

print("Gradient:", gradient)

# Gradient Descent Update
weight = weight - learning_rate * gradient

print("Updated Weight:", weight)

# New Prediction
new_prediction = weight * x

print("New Prediction:", new_prediction)