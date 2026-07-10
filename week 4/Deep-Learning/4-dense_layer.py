"""
What is a Dense Layer?

Until now we had this:

Input
   │
   ▼
Neuron 1

Neuron 2

Neuron 3

Each neuron had its own weights.

Instead of writing every neuron manually...

We group them into a Dense Layer.

A Dense Layer is simply:

A collection of neurons where every input is connected to every neuron.

That's why it's called Dense (Fully Connected).

Visualize It
           Input Layer

      x1      x2      x3
       │ \    │    / │
       │  \   │   /  │
       │   \  │  /   │
       ▼    ▼ ▼ ▼    ▼

      N1   N2   N3   N4

         Dense Layer

Every neuron receives all inputs.
"""


# Step 1 — Import NumPy
import numpy as np

# Step 2 — Create ReLU
def relu(x):
    return np.maximum(0, x)

# Step 3 — Create the Dense Layer Class
# class DenseLayer:

# This class represents one fully connected layer.
# Step 4 — Constructor
# Instead of manually writing weights every time...
# Let's generate them automatically.

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)
    
    
    # Forward Method

    def forward(self, inputs):

        z = np.dot(self.weights, inputs) + self.biases

        output = relu(z)

        return output
    

# Create Input

inputs = np.array([5, 8, 6], dtype=np.float32)

dense = DenseLayer(
    n_inputs=3,
      n_neurons=4)

"""Now something cool happens.

The weights are generated automatically."""

print("Weights:")
print(dense.weights)

# Predict
output = dense.forward(inputs)

print("Output:")
print(output)


