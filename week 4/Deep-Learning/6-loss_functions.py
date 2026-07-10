"""
What is a Loss Function?
A loss function is simply a mathematical function that tells us:
"How wrong is my prediction?"

Smaller loss = Better prediction
Larger loss = Worse prediction

Our goal during training is simple:
Minimize the loss.
"""

# NumPy Implementation

import numpy as np

# Actual values
y_true = np.array(
    [10,20,30,40],
    dtype=np.float32
)

# Predicted values
y_pred = np.array(
    [12,18,33,39],
    dtype=np.float32
)

# Mean Squared Error (MSE)
mse = np.mean(np.square(y_true - y_pred))

print("Mean Squared Error:", mse)

# Mean Absolute Error (MAE)
"""
Mean Absolute Error (MAE)
Instead of squaring...
Take the absolute value.
"""
mae = np.mean(np.abs(y_true - y_pred))

print("Mean Absolute Error:", mae)


"""
MSE vs MAE
MSE	MAE
Squares the errors	Uses absolute errors
Penalizes large mistakes more	Treats all mistakes proportionally
Smooth for optimization	More robust to outliers

For many regression problems, MSE is the default because it works well with gradient-based optimization.
"""