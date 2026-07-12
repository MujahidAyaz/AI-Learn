"""
Before We Start
Imagine you're training a neural network on the ImageNet dataset.
How many images?
≈ 14,000,000 images
Now suppose your computer had to do this:
Read all 14 million images

↓

Calculate predictions

↓

Calculate loss

↓

Update weights

Question...
How much RAM would you need?
A LOT
It would also take a long time before making even a single weight update.
There has to be a better way.
Three Ways to Train a Neural Network

1. Batch Gradient Descent
Uses the entire dataset before updating the weights.

Dataset
────────────────────────────

1000 Samples

↓

One Prediction

↓

One Loss

↓

One Weight Update
Advantages
Stable gradients
Smooth convergence

Disadvantages
Slow
High memory usage
Doesn't scale well

2. Stochastic Gradient Descent (SGD)
Uses one sample at a time.

Sample 1

↓

Update

↓

Sample 2

↓

Update

↓

Sample 3

↓

Update

Advantages
Very fast updates
Low memory

Disadvantages
Noisy learning
Can bounce around the optimum

3. Mini-Batch Gradient Descent ⭐⭐⭐⭐⭐
The best of both worlds.

Instead of
1000 Samples
or
1 Sample

Use
32 Samples
or
64 Samples
or
128 Samples
Now the process becomes

Batch 1

↓

Update

↓

Batch 2

↓

Update

↓

Batch 3

↓

Update

This is what almost every modern deep learning model uses.
"""

# NumPy Example
# We'll use our previous dataset.

import numpy as np
np.random.seed(42)
x = np.array([1,2,3,4,5,6,7,8], dtype=np.float32)
y = np.array([2,4,6,8,10,12,14,16], dtype=np.float32)

# Batch Size

batch_size = 2

# Loop Through Mini-Batches

for i in range(0, len(x), batch_size):

    x_batch = x[i:i+batch_size]

    y_batch = y[i:i+batch_size]

    print("Input Batch:", x_batch)

    print("Target Batch:", y_batch)

    print("-"*30)

# Now Let's Train
# Instead of using the whole dataset...
# We'll use one batch at a time.

weight = np.random.randn()
learning_rate = 0.01
epochs = 100
batch_size = 2

# Training Loop

for epoch in range(epochs):

    for i in range(0, len(x), batch_size):

        x_batch = x[i:i+batch_size]

        y_batch = y[i:i+batch_size]

        prediction = weight * x_batch

        loss = np.mean((y_batch - prediction)**2)

        gradient = (-2/len(x_batch))*np.sum(
            x_batch*(y_batch-prediction)
        )

        weight -= learning_rate*gradient

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} "
            f"Loss {loss:.4f} "
            f"Weight {weight:.4f}"
        )

# Notice something.
#    The weight is updated after every mini-batch, not after the whole dataset.