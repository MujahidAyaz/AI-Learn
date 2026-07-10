# Complete Code


import numpy as np

np.random.seed(42)


def relu(x):
    """ReLU Activation Function"""
    return np.maximum(0, x)


class DenseLayer:
    """Fully Connected Layer"""

    def __init__(self, n_inputs, n_neurons):

        self.weights = np.random.randn(
            n_neurons,
            n_inputs
        )

        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):

        self.output = relu(
            np.dot(
                self.weights,
                inputs
            ) + self.biases
        )


# Input Features
inputs = np.array(
    [5,8,6],
    dtype=np.float32
)

# Layer 1
layer1 = DenseLayer(
    n_inputs=3,
    n_neurons=5
)

# Layer 2
layer2 = DenseLayer(
    n_inputs=5,
    n_neurons=2
)

# Forward Pass
layer1.forward(inputs)

layer2.forward(layer1.output)

print("Layer 1 Output")
print(layer1.output)

print("\nLayer 2 Output")
print(layer2.output)