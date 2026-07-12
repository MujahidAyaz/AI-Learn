"""
A Question

Imagine you're standing on a mountain.
Gradient Descent says:
Take one step downhill.
What if...
The mountain looks like this?

          /\ 
         /  \
        /    \
_______/      \_______

You may:

move slowly,
oscillate,
or get stuck.

Researchers asked:
"Can we design smarter ways to move?"

The answer is optimizers.
What Is an Optimizer?
An optimizer is simply an algorithm that decides:

How big should the next step be?
Should we remember previous steps?
Should every weight move by the same amount?

Gradient Descent answers only the first question in a very basic way.
Modern optimizers answer all three.

Four Optimizers You Must Know
Gradient Descent
        ↓
Momentum
        ↓
RMSProp
        ↓
Adam

These four are enough to understand almost every modern deep learning project.

1. Gradient Descent
We already know it.

Weight

↓

Gradient

↓

Update

Problem:
If gradients keep changing direction...
The weight can zig-zag.

2. Momentum

Think of pushing a heavy shopping cart.
When you push once...
It doesn't stop immediately.
It keeps rolling.
Momentum does the same thing.
Instead of using only today's gradient...
It remembers yesterday's movement.

Yesterday

↓

Today

↓

Combined Motion

This reduces oscillation and speeds up learning.
Visual
Without Momentum

↘
 ↗
  ↘
   ↗

With Momentum
↓↓↓↓↓↓
Much smoother.
"""

# Tiny NumPy Example
"""
velocity = 0
momentum = 0.9
gradient = 5
velocity = momentum * velocity + gradient
weight -= learning_rate * velocity


# Notice the new variable: velocity
# The optimizer now has a memory of previous updates.
"""



"""
3. RMSProp

Now imagine two weights:

Weight A
Gradient = 100
Weight B
Gradient = 0.01

Should they move by exactly the same learning rate?
Probably not.
RMSProp adjusts the learning rate for each parameter individually based on the history of its squared gradients.
Large gradients get smaller effective updates.
Small gradients get relatively larger ones.
"""


"""
4. Adam (Adaptive Moment Estimation)

Adam combines:

Momentum ✔
RMSProp ✔

That's why it's one of the most popular optimizers.
It remembers previous gradients and adapts learning rates automatically.
In many real-world projects, Adam is a great default choice.
Why Is Adam So Popular?
Because it often:

converges faster,
handles noisy gradients well,
requires less manual tuning than plain gradient descent.

It's not always the best choice for every problem, but it's an excellent starting point.

"""

# Simple NumPy Demonstration
import numpy as np

learning_rate = 0.1
weight = 5.0

gradients = [5, 4, 3, 2, 1]

print("Gradient Descent")

for grad in gradients:
    weight -= learning_rate * grad
    print(f"Weight: {weight:.2f}")