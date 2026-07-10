"""
The position parameter is updated during each iteration using the following equation:

    x_new = x_old - learning_rate * gradient
 x_new is the updated position,
 x_old is the current position, 
 learning_rate is a hyperparameter that controls the step size, 
 and gradient is the derivative of the function at the current position.

 Why Subtract the Gradient?

Remember:
    Positive gradient → moving right increases the loss.
    Negative gradient → moving right decreases the loss.

So we always move in the opposite direction of the gradient.
That's why we subtract it.
"""


import numpy as np

#initial weights
weights = 0.8

# Gradient
gradient = 5

# Learning rate
learning_rate = 0.1

# Gradient Descent Update
new_weights = weights - learning_rate * gradient

print("Old Weights:", weights)
print("Gradient:", gradient)
print("Learning Rate:", learning_rate)
print("New Weights:", new_weights)

"""
Let's Simulate Multiple Updates
Instead of updating once...
Let's update several times.
"""
weights = 5
learning_rate = 0.1

for epoch in range(5):
    gradient = 2 * weights  # Derivative of the function f(x) = x^2 is f'(x) = 2x
    weights = weights - learning_rate * gradient
    print(f"Iteration {epoch+1}: Weights = {weights:.4f}")


"""
Notice something.

The weight keeps moving closer to:
0

Why?

Because our example function is:
f(w)=w^2

Its minimum is at:
w=0
Gradient descent is finding that minimum automatically.
"""