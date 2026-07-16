"""
Loss Functions

Topics Covered
--------------
1. What is a Loss Function?
2. Mean Squared Error (MSE Loss)
3. Cross Entropy Loss
4. Binary Cross Entropy Loss
5. Regression vs Classification
6. Choosing the Correct Loss Function

"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
import torch.nn as nn

print("=" * 70)
print("PYTORCH LOSS FUNCTIONS")
print("=" * 70)

# ==========================================================
# PART 1 - Mean Squared Error Loss (Regression)
# ==========================================================

print("\n1. Mean Squared Error (MSE Loss)")
print("-" * 70)

"""
Imagine we are predicting house prices.

Actual Prices:
100, 200, 300

Predicted Prices:
110, 180, 290
"""

predictions = torch.tensor([110.0, 180.0, 290.0])
targets = torch.tensor([100.0, 200.0, 300.0])

mse_loss = nn.MSELoss()

loss = mse_loss(predictions, targets)

print("Predictions :", predictions)
print("Targets     :", targets)
print("MSE Loss    :", loss.item())

# ==========================================================
# PART 2 - Binary Cross Entropy Loss
# ==========================================================

print("\n2. Binary Cross Entropy Loss")
print("-" * 70)

"""
Binary Classification

0 = Not Spam
1 = Spam

The model outputs probabilities.
"""

predictions = torch.tensor([0.95, 0.20, 0.80])

targets = torch.tensor([1.0, 0.0, 1.0])

bce_loss = nn.BCELoss()

loss = bce_loss(predictions, targets)

print("Predictions :", predictions)
print("Targets     :", targets)
print("BCE Loss    :", loss.item())

# ==========================================================
# PART 3 - Cross Entropy Loss
# ==========================================================

print("\n3. Cross Entropy Loss")
print("-" * 70)

"""
Three Classes

0 = Cat
1 = Dog
2 = Horse

Notice:
We DO NOT apply Softmax ourselves.

CrossEntropyLoss performs LogSoftmax internally.
"""

scores = torch.tensor([
    [2.5, 1.0, 0.2],
    [0.1, 3.0, 0.5]
])

targets = torch.tensor([0, 1])

cross_entropy = nn.CrossEntropyLoss()

loss = cross_entropy(scores, targets)

print("Raw Scores")
print(scores)

print("\nCorrect Labels")
print(targets)

print("\nCross Entropy Loss")
print(loss.item())

# ==========================================================
# PART 4 - Good Predictions vs Bad Predictions
# ==========================================================

print("\n4. Good vs Bad Predictions")
print("-" * 70)

good_predictions = torch.tensor([0.98, 0.02, 0.97])
bad_predictions = torch.tensor([0.55, 0.45, 0.40])

targets = torch.tensor([1.0, 0.0, 1.0])

loss_good = nn.BCELoss()(good_predictions, targets)
loss_bad = nn.BCELoss()(bad_predictions, targets)

print("Good Prediction Loss :", loss_good.item())
print("Bad Prediction Loss  :", loss_bad.item())

# ==========================================================
# PART 5 - Summary
# ==========================================================

print("\n" + "=" * 70)
print("WHEN SHOULD YOU USE EACH LOSS FUNCTION?")
print("=" * 70)

print("""
Regression
----------
Task:
Predict continuous values

Examples:
- House Price
- Temperature
- Salary

Use:
✔ nn.MSELoss()

------------------------------------------------------------

Binary Classification
---------------------
Task:
Two classes

Examples:
- Spam / Not Spam
- Fraud / Not Fraud
- Pass / Fail

Use:
✔ nn.BCELoss()
or
✔ nn.BCEWithLogitsLoss() (Preferred)

------------------------------------------------------------

Multi-Class Classification
--------------------------
Task:
Three or more classes

Examples:
- Cat
- Dog
- Horse

Use:
✔ nn.CrossEntropyLoss()

IMPORTANT:
Do NOT apply Softmax before CrossEntropyLoss.
""")

print("=" * 70)
print("Lesson Completed Successfully!")
print("=" * 70)