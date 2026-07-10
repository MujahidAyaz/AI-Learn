"""
Until now we had one Dense Layer.

Input
   │
Dense Layer
   │
Output

Today we're going to connect multiple layers.

Input
   │
Dense Layer 1
   │
ReLU
   │
Dense Layer 2
   │
Output
"""

"""
Before Coding

Let's understand WHY we need multiple layers.
Imagine you're building an AI that recognizes cats.

The first layer may learn:

Edges

↓

The second layer combines edges into

Eyes

Ears

Nose

↓

The third layer combines them into

Cat Face

↓

Final Layer

Cat

Notice something.
Every layer learns more abstract features.
This is one of the biggest ideas in Deep Learning."""


import numpy as np

def relu(x):
    """ReLU Activation Function"""
    return np.maximum(0, x)

# Dense Layer
# Instead of returning the output...
# We'll save it inside the object.

class DenseLayer:

    """A Fully Connected Neural Network Layer"""

    def __init__(self, n_inputs, n_neurons):

        self.weights = np.random.randn(n_neurons, n_inputs)

        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):

        self.output = relu(np.dot(self.weights, inputs) + self.biases)

"""
Notice

We store
self.output

instead of
return output

This is exactly how many deep learning libraries work."""

# Input

inputs = np.array([5, 8, 6], dtype=np.float32)

layer1 = DenseLayer(
    n_inputs=3,
    n_neurons=5
)

# Forward Pass
layer1.forward(inputs)
#now layer1.output contains the output of the first layer.
layer1.output

"""
Second Layer
What should the number of inputs be?

Think...
Layer 1 produced
5 outputs

Therefore
Layer 2 receives
5 inputs
Exactly.
"""

layer2 = DenseLayer(
    n_inputs=5,
    n_neurons=2
)
layer2.forward(layer1.output)

layer2.output
print(f'layer1.output: {layer1.output}')

print(f'layer2.output: {layer2.output}')