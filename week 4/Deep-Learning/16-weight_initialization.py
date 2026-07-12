"""
The Problem
Suppose we build a neural network with 100 neurons.
Question:
What should the initial weights be?

Option A
All weights = 0
Option B
Random values

Which one should we choose?
Why Not Initialize Everything to Zero?
Imagine this hidden layer:

Input

↓

Neuron 1

↓

Neuron 2

↓

Neuron 3

If every neuron starts with:
Weight = 0
Bias = 0

What happens?
Each neuron computes exactly the same output.
Example:

Neuron 1

Output = 0

Neuron 2

Output = 0

Neuron 3

Output = 0

During backpropagation...
They all receive the same gradients.
After updating:

Neuron 1

0.05

Neuron 2

0.05

Neuron 3

0.05

Still identical.
Next epoch...
Still identical.
They never specialize.
This Is Called the Symmetry Problem
If neurons start identically...
They stay identical.
The network behaves as if there were only one neuron.
Instead of learning different features...
Every neuron learns the same thing.
Random Initialization Solves This
Instead of:

0
0
0

Use:

0.31
-0.18
0.07

Now every neuron starts differently.
Because their outputs differ...
Their gradients differ...
They gradually learn different patterns.
"""

# NumPy Example
import numpy as np

np.random.seed(42)

weights = np.random.randn(5)

print(weights)

"""
But There's Another Problem...
What if we initialize weights like this?
500
-800
300
900

Or:

0.000000001
-0.000000003

Both are random...
Yet both are terrible.
Why?
Because the scale of the initial weights matters.

Very Large Weights

Suppose:
Input = 10
Weight = 500

Output: 5000
After passing through a sigmoid: ≈ 1.0

The neuron saturates.
Its gradient becomes almost zero.
Learning slows dramatically.

Very Small Weights

Suppose:
Weight = 0.00000001
Output:   ≈ 0

After several layers...
The signals become tiny.
This contributes to the vanishing gradient problem, where updates become extremely small.
So What's the Solution?
Researchers designed smarter initialization methods.
The two you'll see most often are:

Xavier (Glorot) Initialization
He Initialization

"""
# Xavier Initialization

# Best suited for:
# Sigmoid
# Tanh
# It keeps the variance of activations roughly stable across layers.
# In NumPy:

fan_in = 4
fan_out = 3

limit = np.sqrt(6 / (fan_in + fan_out))

weights = np.random.uniform(
    -limit,
    limit,
    (fan_in, fan_out)
)

print(weights)

"""
He Initialization

Best suited for:

ReLU
Leaky ReLU
"""
# NumPy:

fan_in = 4
fan_out = 3

weights = np.random.randn(
    fan_in,
    fan_out
) * np.sqrt(2 / fan_in)

print(weights)

# This is the default initialization used for many ReLU-based networks because it helps preserve signal magnitude through the layers.
