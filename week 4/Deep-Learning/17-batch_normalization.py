"""
Almost every modern neural network uses some form of normalization.
Imagine a Classroom
Suppose a teacher gives a math exam.

Student A scores:
20
Student B scores:
95
Student C scores:
60

Now imagine another class where the exam was much harder.

Scores:

5
15
10

Can we compare these scores fairly?
Not really.
The scales are different.
We first normalize them.
Deep learning does the same thing.

The Problem

Suppose Layer 1 outputs:
[2, 4, 6, 8]
Next batch:
[200, 400, 600, 800]

Next layer now has to learn from wildly different input scales.
Training becomes unstable.

### Internal Covariate Shift

As earlier layers learn, the distribution of their outputs changes.
That means deeper layers are constantly receiving new kinds of inputs.
It's like trying to learn while the textbook changes every day.
Batch Normalization reduces this instability by normalizing activations within each mini-batch.


The Idea

Given a batch:

[2, 4, 6, 8]

Compute:

Mean
μ = 2+4+6+8/4  = 5

Standard Deviation
We'll let NumPy compute it.
Normalize
        xnorm = x − μ/σ + ϵ 
where:

μ = batch mean
σ = batch standard deviation
ϵ = very small number to avoid division by zero

After normalization, the batch has:
Mean ≈ 0
Standard Deviation ≈ 1
"""
# NumPy Example

import numpy as np

# Sample Batch
x = np.array([2, 4, 6, 8], dtype=np.float32)

# Statistics
mean = np.mean(x)
std = np.std(x)

# Normalize
x_norm = (x - mean) / (std + 1e-8)

print("Original Batch :", x)
print("Mean           :", mean)
print("Std            :", std)
print("Normalized     :", x_norm)
print("New Mean       :", np.mean(x_norm))
print("New Std        :", np.std(x_norm))

"""
But Wait...
If everything becomes centered at zero...
Did we lose information?
No.

BatchNorm introduces two learnable parameters:

y=γxnorm ​+ β

where:

γ (gamma) controls the scale.
β (beta) controls the shift.

The network can learn the most useful distribution during training.
Why Batch Normalization Helps

It often:

Speeds up convergence.
Stabilizes training.
Allows higher learning rates.
Acts as a mild regularizer.
Helps gradients flow through deep networks.
"""
