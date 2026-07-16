"""


Topics Covered
--------------
1. ReLU
2. Sigmoid
3. Tanh
4. Softmax
5. Real World Example
"""

import torch
import torch.nn as nn

# ----------------------------------------------------
# Create Sample Input
# ----------------------------------------------------

x = torch.tensor([-3., -2., -1., 0., 1., 2., 3.])

print("=" * 60)
print("Original Tensor")
print("=" * 60)
print(x)

# ----------------------------------------------------
# ReLU Activation
# ----------------------------------------------------

relu = nn.ReLU()

relu_output = relu(x)

print("\nReLU Output")
print(relu_output)

# ----------------------------------------------------
# Sigmoid Activation
# ----------------------------------------------------

sigmoid = nn.Sigmoid()

sigmoid_output = sigmoid(x)

print("\nSigmoid Output")
print(sigmoid_output)

# ----------------------------------------------------
# Tanh Activation
# ----------------------------------------------------

tanh = nn.Tanh()

tanh_output = tanh(x)

print("\nTanh Output")
print(tanh_output)

# ----------------------------------------------------
# Softmax Example
# ----------------------------------------------------

scores = torch.tensor([[2.0, 1.0, 0.5]])

softmax = nn.Softmax(dim=1)

probabilities = softmax(scores)

print("\nSoftmax Probabilities")
print(probabilities)

print("\nSum of Probabilities")
print(probabilities.sum())

# ----------------------------------------------------
# Real World Example
# ----------------------------------------------------

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

sample = torch.rand(2, 4)

prediction = model(sample)

print("\nModel Output")
print(prediction)

print("\nOutput Shape")
print(prediction.shape)

print("\nLesson Completed Successfully!")