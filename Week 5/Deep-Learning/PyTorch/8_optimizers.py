"""
 Optimizers

Topics Covered
--------------
1. What is an Optimizer?
2. Stochastic Gradient Descent (SGD)
3. Adam Optimizer
4. Learning Rate
5. optimizer.step()
6. optimizer.zero_grad()
"""

# ==========================================================
# Import Libraries
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# Create Dummy Dataset
# ==========================================================

"""
Suppose we have a simple regression problem.

Input  -> Study Hours
Output -> Exam Marks
"""

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [20.0],
    [40.0],
    [60.0],
    [80.0],
    [100.0]
])

# ==========================================================
# Create Neural Network
# ==========================================================

model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.MSELoss()

# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

# ==========================================================
# Training Loop
# ==========================================================

epochs = 10

print("=" * 60)
print("Training Started")
print("=" * 60)

for epoch in range(epochs):

    
    # Forward Pass
    predictions = model(X)

   
    # Calculate Loss
    loss = criterion(predictions, y)


    # Clear Previous Gradients
    optimizer.zero_grad()


    # Backpropagation
    loss.backward()

    # Update Weights
    optimizer.step()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {loss.item():.4f}"
    )

print("\nTraining Finished Successfully!")

# ==========================================================
# Test the Model
# ==========================================================

test_data = torch.tensor([[6.0]])

prediction = model(test_data)

print("\nPrediction for 6 Study Hours:")
print(prediction)