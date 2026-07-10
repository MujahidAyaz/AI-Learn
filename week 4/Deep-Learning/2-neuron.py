import numpy as np

# Create the Inputs
# Suppose we're predicting whether a student will pass.

# Features:

# Hours Studied
# Attendance
# Assignments Completed

inputs = np.array([5, 8, 6], dtype=np.float32)


"""Initialize the Weights

For now, we'll choose the weights manually.

Later, the network will learn them.
"""

weights = np.array([0.4, 0.7, 0.2], dtype=np.float32)

# Add the Bias
bias = 1.0

# Calculate the Weighted Sum
z = np.dot(inputs, weights) + bias

print("Weighted Sum:", z)



# Create ReLU
# Instead of using NumPy directly every time:

np.maximum(0, z)


# Let's build our own function.

def relu(x):
    return np.maximum(0, x)

# Now use it.

output = relu(z)

print("Neuron Output:", output)


#  Test Negative Inputs

# Now let's see ReLU in action.

# Change the weights.

weights = np.array([-2, -1, -0.5], dtype=np.float32)