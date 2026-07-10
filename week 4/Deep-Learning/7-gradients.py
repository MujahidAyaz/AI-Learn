
print("==" * 50)

# NumPy Example

import numpy as np

def f(x):
    return x**2


x=3.0
h=0.001

gradient = (f(x+h)-f(x-h))/(2*h)
print("Gradient at x=3:", gradient)


print("==" * 50)
#==============================================================================================

# Complete Code

import numpy as np

def square_function(x):
    """Returns x squared."""
    return x ** 2

x = 3.0
h = 0.001

gradient = (square_function(x + h) - square_function(x)) / h

print("x:", x)
print("f(x):", square_function(x))
print("Approximate Gradient:", gradient)


print("==" * 50)

