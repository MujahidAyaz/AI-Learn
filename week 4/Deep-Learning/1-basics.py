import numpy as np


# Scalars
# A scalar is a single number.

Learning_rate=0.1
print("Learning rate is set to:", Learning_rate)
print(type(Learning_rate))
"""
Examples in Deep Learning

Bias
Learning Rate
Loss
Accuracy

These are often scalars.

"""

# Vectors
# A vector is a 1D array.

student = np.array([5, 8, 6])

print(student)


# Matrix

# Now suppose we have 4 students.

students = np.array([
    [5, 8, 6],
    [7, 9, 9],
    [3, 6, 5],
    [9, 10, 10]
])

print(students)


# Shape
# This is one of the most used methods in Deep Learning.

print(students.shape)

# Number of Dimensions
print(student.ndim)

print(students.ndim)


student = student.astype(np.float32)
#Why?
#Because deep learning models usually operate with floating-point numbers.




# Dot Product
# Now we're getting closer to a neuron.

# Input
x = np.array([5, 8, 6])

# Weights
w = np.array([0.4, 0.7, 0.2])

# Now calculate
z = np.dot(x, w)

print(z)



# Add Bias
bias = 1
z = np.dot(x, w) + bias
print(z)

