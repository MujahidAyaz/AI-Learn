"""
What is torch.nn?
torch.nn is a module containing everything required to build neural networks.
It includes:
Layers
Activation Functions
Loss Functions
Containers
Neural Network Utilities
"""
import torch
import torch.nn as nn

layer = nn.Linear(
    in_features=3,
    out_features=2
)

#Inspecting the Entire Layer
print(layer)
print("-"*20)
# printing other things included in layer
print(layer.weight)
print("-"*20)
print(layer.weight.shape)
print("-"*20)
print(layer.bias)
print("-"*20)
print(layer.bias.shape)
print("-"*20)

# Feeding Data into the Layer
# Create input
x = torch.tensor([
    25.0,
    50000.0,
    3.0
])

# Pass it
output = layer(x)
print(output)
print("-"*20)
#Batch Input
#Real neural networks don't process one sample.
#They process batches.
#Example

x = torch.tensor([
    [25.0,50000.0,3.0],
    [30.0,62000.0,5.0],
    [22.0,35000.0,1.0]
])

#Pass through layer
output = layer(x)
print(output)
print("-"*20)
#Output shape
print(output.shape)
print("-"*20)
# Parameters
#A neural network learns by updating parameters.

for param in layer.parameters():
    print(param)
print("-"*20)
# Counting Parameters
for name, param in layer.named_parameters():

    print(name)
    print(param.shape)
print("-"*20)

"""
Summary
Concept	Meaning
torch.nn	Neural network module
nn.Linear	Fully connected layer
in_features	Number of input features
out_features	Number of output neurons
layer.weight	Trainable weights
layer.bias	Trainable biases
layer.parameters()	Returns all trainable parameters
named_parameters()	Returns parameter names and values
"""