# Complete code for gradient descent algorithm

import numpy as np

# Initial Weight
weight = 5.0

# Learning Rate
learning_rate = 0.1

print("Starting Weight:", weight)
print("-" * 35)

# Gradient Descent Loop 
for epoch in range(10):

    # Gradient of f(w) = w² 
    gradient = 2 * weight

    # Update Rule 
    weight = weight - learning_rate * gradient 

    print(
        f"Epoch {epoch+1:2d} | " # Print epoch number
        f"Gradient: {gradient:7.4f} | "  # Print gradient value
        f"Learning Rate: {learning_rate:7.4f} | "  # Print learning rate
        f"Weight: {weight:7.4f}" # Print updated weight
    )