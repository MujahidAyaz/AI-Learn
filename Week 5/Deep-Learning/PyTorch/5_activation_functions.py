# PyTorch Activation Functions
import torch
import torch.nn as nn

# Some common activation functions are:


import torch
import torch.nn as nn

#nn.ReLU()
relu = nn.ReLU()
x = torch.tensor([-3., -1., 0., 2., 5.])
print(relu(x))
"""
Why is ReLU Popular?

Advantages:
Very fast
Easy to compute
Helps deep networks train efficiently
Reduces vanishing gradient problems

Today, ReLU is the default activation in most hidden layers.
"""

#nn.Sigmoid()

sigmoid = nn.Sigmoid()
x = torch.tensor([-3., 0., 3.])
print(sigmoid(x))
"""
When is Sigmoid Used?

Mostly:
Binary Classification
Probability outputs

Problem with Sigmoid
When values become very large or very small:
Gradient
↓
Almost Zero
This causes the Vanishing Gradient Problem, making deep networks hard to train.
For this reason, Sigmoid is rarely used in hidden layers today.
"""
#nn.Tanh()

tanh = nn.Tanh()
x = torch.tensor([-3., 0., 3.])
print(tanh(x))
"""
Why Tanh Was Popular
Unlike Sigmoid:
Centered around 0
This often helps optimization.
However, it can still suffer from vanishing gradients.
"""

#nn.Softmax(dim=1)
# Softmax converts raw scores into probabilities that sum to 1.
# Example:

softmax = nn.Softmax(dim=1)
scores = torch.tensor([[2.0, 1.0, 0.1]])
print(softmax(scores))

"""
Which Activation Should You Use?

For hidden layers:
✅ ReLU

For binary classification output:
✅ Sigmoid

For multi-class classification output:
✅ Softmax

Tanh is less common today but still appears in some architectures.
"""

# Mini Project
print("----------Mini Project----------")
import torch
import torch.nn as nn

x = torch.tensor([-2., -1., 0., 1., 2.])

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

print("Original :", x)
print("ReLU     :", relu(x))
print("Sigmoid  :", sigmoid(x))
print("Tanh     :", tanh(x))

# Then create:
scores = torch.tensor([[3.0, 1.0, 0.2]])
softmax = nn.Softmax(dim=1)
print(softmax(scores))