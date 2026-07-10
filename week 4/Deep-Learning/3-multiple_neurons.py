import numpy as np

def relu(x):
    """ReLU Activation Function"""
    return np.maximum(0,x)


# Input Features
inputs = np.array([5,9,7], dtype=np.float32)

# Weight Matrix
weights= np.array(
    [
        [0.4, 0.7, 0.2],
        [0.1, 0.9, 0.5],
        [0.8, 0.2, 0.6]
    ], dtype=np.float32)

# Bias for Each Neuron
biases = np.array ([1,0.5,2], dtype=np.float32)

# Forward Pass
z=np.dot(weights, inputs) + biases

# Apply ReLU Activation Function
output = relu(z)

print("\nAfter ReLU:")
print(output)