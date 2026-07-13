"""
Imagine a Football Team ⚽
Suppose a football team has one superstar player.
Every match, everyone passes the ball only to him.
If that player gets injured...
The whole team collapses.
That team has become overdependent on one player.
Neural networks can develop the same problem.
Certain neurons become so important that the network relies almost entirely on them.

Dropout prevents this.
The Idea
During training...
Randomly turn off some neurons.

Example:

Layer Before
● ● ● ● ● ● ● ●

After applying Dropout (rate = 0.5)
● ✖ ● ✖ ● ✖ ✖ ●

The crossed-out neurons contribute nothing during that forward pass.
Next mini-batch?
A different random set is dropped.
Why Does This Help?
Because the network can never assume: "Neuron #5 will always be available."
Every neuron must learn useful features independently.
This encourages the network to build redundant and robust representations instead of depending on a few specific neurons.

# Training vs Inference

A common interview question:
Do we use Dropout during inference?
Answer: No.

Training:

Neuron 1 ✅
Neuron 2 ❌
Neuron 3 ✅
Neuron 4 ❌

Inference:

Neuron 1 ✅
Neuron 2 ✅
Neuron 3 ✅
Neuron 4 ✅

We only inject randomness while learning.
When making predictions, we want the full network.
Dropout Rate

Typical values:

Layer	Common Rate
Hidden Layers	0.2 – 0.5
Input Layer	Usually none or very small
Output Layer	Almost never
"""

# Numpy Example

import numpy as np

np.random.seed(42)

x= np.array([1,2,3,4,5], dtype=np.float32)

dropout_rate = 0.4

mask = np.random.rand(len(x)) > dropout_rate

print ("mask :",mask)

output = x * mask
print("Original X :",x)
print("After Dropout :",output)

"""
Notice:

The neurons weren't deleted.
They were only ignored for this training step.

# Inverted Dropout

Modern frameworks don't just set neurons to zero.
They also scale the remaining activations.

Formula:  output= mask * x/1 - p

where:

p = dropout rate

Why?

Suppose:
Dropout = 0.5
Without scaling:
Average activation becomes roughly half.
The next layer would receive much smaller values during training than during inference.
By dividing by 1−p, we keep the expected activation level approximately the same in both phases.
PyTorch uses this "inverted dropout" approach automatically.
"""
# NumPy Example (Inverted Dropout)
print("-"*30)
print("Inverted Dropout : ")
print("-"*30)


import numpy as np

np.random.seed(42)

x = np.array([1, 2, 3, 4, 5], dtype=np.float32)

dropout_rate = 0.4

mask = (np.random.rand(len(x)) > dropout_rate).astype(np.float32)

output = x * mask / (1 - dropout_rate)

print("Original :", x)
print("Mask     :", mask)
print("Output   :", output)