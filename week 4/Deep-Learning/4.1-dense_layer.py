import numpy as np


def relu(x):
    """ReLU Activation Function"""
    return np.maximum(0, x)


class DenseLayer:
    """A Fully Connected Neural Network Layer"""

    def __init__(self, n_inputs, n_neurons):

        self.weights = np.random.randn(n_neurons, n_inputs)

        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):

        z = np.dot(self.weights, inputs) + self.biases

        output = relu(z)

        return output


inputs = np.array([5, 8, 6], dtype=np.float32)

dense = DenseLayer(
    n_inputs=3,
    n_neurons=4
)

print("Weights:")
print(dense.weights)

print("\nBiases:")
print(dense.biases)

output = dense.forward(inputs)

print("\nOutput:")
print(output)