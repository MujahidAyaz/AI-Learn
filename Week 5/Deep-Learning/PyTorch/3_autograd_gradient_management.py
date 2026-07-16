# Gradient Accumulation
# Let's start with this code.

import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)

# Output
# tensor(4.)
# Everything looks fine.
# Now let's run backward() again.

import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
y.backward()
print(x.grad)

"""
What do you expect?
Most beginners answer:
4
Actual output
tensor(8.)
Wait...Why?
Why Did It Become 8?
PyTorch adds gradients instead of replacing them.
First backward
Gradient = 4
Second backward
Gradient += 4
↓
8
Third backward
12
Fourth backward
16
PyTorch accumulates gradients.
Why Does PyTorch Do This?
Because sometimes we actually want to accumulate gradients.
Suppose your GPU cannot fit a batch of 512 images.
Instead:

Batch
512

You split it into

128
128
128
128

Run

Forward
↓
Backward

four times.
Then update weights once.

This is called:
Gradient Accumulation
It allows you to simulate a larger batch size without needing more GPU memory.
So accumulation is a feature, not a bug.
"""

# zero_grad()
# Since gradients accumulate, we usually want to clear them before the next training step.
# Example:

import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)

#Output
#4

#Now clear the gradient:

x.grad.zero_()
print(x.grad)

#Output
#tensor(0.)

"""
Notice the underscore:
zero_()
In PyTorch, methods ending with _ modify the tensor in place.
Examples include:

zero_()
add_()
fill_()
In Real Neural Networks

You won't usually call:
x.grad.zero_()

Instead you'll use:
optimizer.zero_grad()

Every professional PyTorch training loop starts with:
optimizer.zero_grad()
loss.backward()
optimizer.step()
"""

#detach()
#Suppose:

x = torch.tensor(5.0, requires_grad=True)
y = x * 2
print(y)

#Output
#tensor(10., grad_fn=<MulBackward0>)
#Now detach it.
z = y.detach()
print(z)

#Output
#tensor(10.)
#Notice something missing?
#grad_fn
#It's gone.

"""
Why Use detach()?
Imagine you're evaluating a model.
You don't want gradients.
You're only making predictions.
Detaching saves memory and computation.
It's also useful when converting tensors to NumPy arrays.
"""

# torch.no_grad()
# This is one of the most commonly used contexts in PyTorch.
# Example:

import torch
x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = x * 5
print(y)

#Output
#tensor(10.)
#Notice:
# No grad_fn.

"""
What Happened?
Inside:
with torch.no_grad():
PyTorch stops building the computation graph.
No graph means:
Less memory usage
Faster execution
No gradient computation
When Do We Use torch.no_grad()?
Almost always during:
Model evaluation
Validation
Testing
Inference (making predictions)
Training:
Gradients
YES
Inference:
Gradients
NO
Real Example
Training:
model.train()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
Evaluation:
model.eval()
with torch.no_grad():
    outputs = model(images)
This is standard practice in professional PyTorch code.
"""

"""
Summary
Concept	                            Purpose
Gradient Accumulation	            Gradients are added by default
zero_grad()	                        Clears gradients before the next update
detach()	                        Stops tracking a specific tensor
torch.no_grad()	                    Disables gradient tracking within a block
"""