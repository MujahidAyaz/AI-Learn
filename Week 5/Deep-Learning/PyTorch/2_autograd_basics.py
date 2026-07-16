import torch


# Create a Tensor That Tracks Gradients
x = torch.tensor(2.0, requires_grad=True)
print(x)
"""
Notice:
requires_grad=True
This tells PyTorch:
"Track every operation involving this tensor."
"""


# Perform Operations
y = x ** 2
print(y)
"""
New thing:
grad_fn
What is that?
It means:
This tensor was created by an operation that Autograd is tracking.
"""

# Another Example

x = torch.tensor(3.0, requires_grad=True)
y = 5 * x + 2
y.backward()
print(x.grad)

# More Complex Example
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print(x.grad)

"""Output:
tensor(14.)

Let's verify.
y=x^3+2x
Derivative:
3x^2+2
Substitute:
x = 2
3 × 4 + 2
12 + 2
14
"""


# grad_fn
# Example:

x = torch.tensor(2.0, requires_grad=True)
y = x + 3
print(y.grad_fn)

# Output:
# <AddBackward0>
# This tells us:
# y
# was created by an addition operation.

# Similarly:
y = x * 5
# might show:
# <MulBackward0>
# PyTorch remembers the operation so it can apply the correct derivative during backpropagation.


"""
Leaf Tensors
This term appears often.
A leaf tensor is a tensor created directly by you.

Example:
x = torch.tensor(2.0, requires_grad=True)
x is a leaf tensor.
But:
y = x * 3
y is not a leaf tensor.
Why does it matter?
Because gradients are stored by default only for leaf tensors.
That's why:
print(x.grad)
works after backward(), but y.grad will be None unless you explicitly retain it.
"""


"""
Important Rule
You can only call:
backward()
on a scalar (a tensor containing a single value), unless you provide an explicit gradient.

For example:
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward()

This raises an error because y has three elements.
Instead:

loss = y.sum()
loss.backward()

Now loss is a scalar, and gradients can be computed.
"""
"""
Summary
Concept                     	Meaning
requires_grad=True          	Tell PyTorch to track operations
grad_fn	                        Records how a tensor was created
backward()	                    Computes gradients
.grad	                        Stores the computed gradient
Leaf tensor	                    Tensor created directly by the user
"""