"""
 Creating Your First Neural Network

Topics Covered
--------------
1. nn.Module
2. __init__()
3. super().__init__()
4. nn.Linear()
5. forward()
6. Creating a Model
7. Passing Data
8. Model Parameters
9. Model Summary

============================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
import torch.nn as nn

# ==========================================================
# Create a Custom Neural Network
# ==========================================================


class SimpleNeuralNetwork(nn.Module):
    """
    A simple neural network with:

    Input Layer  -> 4 Features
    Hidden Layer -> 8 Neurons
    Output Layer -> 2 Neurons
    """

    def __init__(self):
        """
        Constructor

        This method runs automatically when
        the model object is created.
        """

        # Initialize the parent nn.Module class
        super().__init__()

        # First Fully Connected Layer
        self.fc1 = nn.Linear(
            in_features=4,
            out_features=8
        )

        # ReLU Activation Function
        self.relu = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(
            in_features=8,
            out_features=2
        )

    def forward(self, x):
        """
        Defines how data flows through the network.
        """

        # Input -> Hidden Layer
        x = self.fc1(x)

        # Apply Activation
        x = self.relu(x)

        # Hidden -> Output
        x = self.fc2(x)

        return x


# ==========================================================
# Create Model
# ==========================================================

model = SimpleNeuralNetwork()

print("=" * 60)
print("Neural Network Architecture")
print("=" * 60)

print(model)

# ==========================================================
# Create Dummy Input Data
# ==========================================================

"""
Suppose we have 3 students.

Each student has 4 features.

Age
Study Hours
Attendance
Assignments
"""

sample_data = torch.tensor([
    [20.0, 6.0, 90.0, 8.0],
    [22.0, 5.0, 85.0, 7.0],
    [19.0, 8.0, 95.0, 9.0]
])

print("\nInput Shape")
print(sample_data.shape)

# ==========================================================
# Forward Pass
# ==========================================================

output = model(sample_data)

print("\nModel Output")
print(output)

print("\nOutput Shape")
print(output.shape)

# ==========================================================
# Model Parameters
# ==========================================================

print("\n" + "=" * 60)
print("Trainable Parameters")
print("=" * 60)

for name, parameter in model.named_parameters():

    print(f"\n{name}")

    print(f"Shape : {parameter.shape}")

    print(parameter)

# ==========================================================
# Count Total Parameters
# ==========================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nTotal Trainable Parameters")
print(total_parameters)

print("\nLesson Completed Successfully!")