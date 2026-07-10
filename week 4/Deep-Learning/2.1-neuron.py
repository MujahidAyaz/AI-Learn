# Complete Code

import numpy as np

def relu(x):
    """ReLU Activation Function"""

    return np.maximum(0, x)


def neuron_output(inputs, weights, bias):

    """Calculate the output of a Single Artificial Neuron given inputs, weights, and bias"""

    # Calculate the Weighted Sum
    z = np.dot(inputs, weights) + bias

    # Apply ReLU Activation Function
    output = relu(z)

    return output

inputs = np.array([5, 8, 6], dtype=np.float32)
weights = np.array([0.4, 0.7, 0.2], dtype=np.float32)
bias = 1.0

# Calculate the output of the neuron
prediction = neuron_output(inputs, weights, bias)
print(prediction)

