#Complete Code

import numpy as np

# Actual Values
y_true = np.array([10, 20, 30])

# Predicted Values
y_pred = np.array([12, 18, 27])

# Mean Squared Error
mse = np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error
mae = np.mean(np.abs(y_true - y_pred))

print("Actual Values:", y_true)
print("Predicted Values:", y_pred)

print("\nMean Squared Error:", mse)
print("Mean Absolute Error:", mae)
