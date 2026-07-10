"""
Imagine this network:

Input
   │
Dense Layer 1
   │
Dense Layer 2
   │
Prediction

Suppose the prediction is wrong.

Question:
Which weight should change?

Layer 1?
Layer 2?
Both?
How much?

This is exactly what Backpropagation solves.

"""

import numpy as np

# Input
x = 3

# Initial Weight
weight = 2.0

# Actual Target
y_true = 10

# Learning Rate
learning_rate = 0.1

# -------------------------
# Forward Pass
# -------------------------

prediction = weight * x

loss = (y_true - prediction) ** 2

print("Prediction:", prediction)
print("Loss:", loss)

# We'll derive that mathematically in the next lesson.
# For now, trust the formula.

gradient = -2 * x * (y_true - prediction)
print(f"Gradient: {gradient}")

"""
Negative gradient.
What does that mean?
Increase the weight.
Exactly what we expected.
"""

# Update Weight
weight = weight - learning_rate * gradient
print(f"Updated Weight: {weight}")

# -------------------------
"""
Did it overshoot?
Yes.
Why?
Learning rate is a bit high.
Exactly why learning rate matters.
"""