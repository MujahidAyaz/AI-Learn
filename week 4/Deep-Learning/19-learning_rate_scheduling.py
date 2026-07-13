"""
Why Learning Rate Scheduling Exists
Suppose you're driving from Islamabad to Murree.
At the beginning:
Road is empty
You can safely drive fast.

Near your destination:

Sharp turns
Traffic
Narrow roads

Would you still drive at 120 km/h?
No.
You slow down.

Deep learning works exactly the same way.
Fixed Learning Rate Problem
Suppose
learning_rate = 0.1

Your weight updates look like

10
↓
8
↓
6
↓
4
↓
2
↓
0
↓
-2
↓
2
↓
-2
↓
2

Notice?
The optimizer keeps overshooting the minimum.
It never settles.
Now suppose

learning_rate = 0.000001

Updates become

10
↓
9.9999
↓
9.9998
↓
9.9997

Training becomes painfully slow.
A single learning rate is rarely perfect.

We want:

Large steps at the beginning.
Small, careful steps near the optimum.

This is called Learning Rate Scheduling.
"""
#-------------------------------------------------------------------------
# 1. Step Decay
# The simplest scheduler.
# Example:

# Epochs 1–10   LR = 0.1
# Epochs 11–20  LR = 0.01
# Epochs 21–30  LR = 0.001

# The learning rate drops at fixed intervals.

# NumPy Example
initial_lr = 0.1
drop_every = 10
drop_factor = 0.1

for epoch in range(1, 31):

    lr = initial_lr * (
        drop_factor ** ((epoch - 1) // drop_every)
    )

    print(
        f"Epoch {epoch:2d} | Learning Rate = {lr:.4f}"
    )

#-------------------------------------------------------------------------

"""
2. Exponential Decay

Instead of sudden drops:

0.100
↓
0.095
↓
0.090
↓
0.086
↓
0.081

The learning rate decreases smoothly every epoch.
Formula

LR=LR0 * e^−kt

where:

LR0 = initial learning rate
k = decay constant
t = epoch number
"""
# NumPy Example

import numpy as np

initial_lr = 0.1
k = 0.1

for epoch in range(10):

    lr = initial_lr * np.exp(-k * epoch)

    print(
        f"Epoch {epoch:2d} | LR = {lr:.5f}"
    )

#-------------------------------------------------------------------------

"""
3. ReduceLROnPlateau ⭐⭐⭐⭐⭐

This is one of the most widely used schedulers.
Instead of reducing the learning rate after a fixed number of epochs, it watches the validation loss.
Example:
Epoch 5
Validation Loss
0.45
↓
Epoch 6
0.43
↓
Epoch 7
0.43
↓
Epoch 8
0.43
↓
Epoch 9
0.43
No improvement.

Scheduler says:
Reduce the learning rate.
This is smarter than Step Decay because it reacts to the model's actual performance.

"""
# Simple Python Logic
best_loss = float("inf")

patience = 3

counter = 0

learning_rate = 0.1

validation_losses = [
    0.90,
    0.70,
    0.50,
    0.45,
    0.45,
    0.45,
    0.45,
]

for loss in validation_losses:

    if loss < best_loss:

        best_loss = loss
        counter = 0

    else:

        counter += 1

        if counter >= patience:

            learning_rate *= 0.1

            print(
                f"Reducing LR to {learning_rate}"
            )

            counter = 0

#-------------------------------------------------------------------------
"""
4. Cosine Annealing

Instead of dropping suddenly...

The learning rate follows a cosine curve.

LR

0.1 ●

    \

     \

      \

       \

0.0────────────

Benefits:

Smooth reduction.
Often finds better minima.
Popular in computer vision and transformer models.

We won't implement it from scratch yet, but you should know where it's used.

Comparison
Scheduler	        Idea	                                     When to Use
Step Decay	        Reduce every fixed number of epochs	         Simple baseline
Exponential Decay	Smooth decrease	                             Stable training
ReduceLROnPlateau	Reduce when validation stops improving	     Most tabular DL projects
Cosine Annealing	Cosine-shaped decay	                         CNNs, Transformers
"""