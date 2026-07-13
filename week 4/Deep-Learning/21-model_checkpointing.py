"""
This lesson is short but extremely important.

Every real deep learning project saves checkpoints.

Why Save Models?

Imagine training a neural network for:
18 hours
At epoch:
97

Your electricity goes out.
Without checkpoints...
You lose everything.
With checkpoints...
You resume from epoch 97.
What Is a Checkpoint?
A checkpoint is simply a snapshot of the training process.
It usually contains:

Model weights
Optimizer state (for continuing training)
Current epoch
Best validation loss
Learning rate (often included)

In our NumPy project, we'll save the weights and metadata.

Save the Best, Not Just the Last
Suppose:

Epoch	Validation Loss
8	0.42
9	0.38
10	0.35 ✅
11	0.37
12	0.41

Should we save Epoch 12?
No.
We save Epoch 10, because it generalized best.
This ties directly to Early Stopping.
"""
# NumPy Example
# We'll use NumPy's save functions.

import numpy as np

weights = np.random.randn(5, 3)

np.save("best_weights.npy", weights)

print("Weights saved successfully.")

#Loading

loaded_weights = np.load("best_weights.npy")
print(loaded_weights)

#Saving Multiple Parameters
#A neural network has more than one weight matrix.
#We can store everything together.

W1 = np.random.randn(10, 5)
b1 = np.random.randn(5)
W2 = np.random.randn(5, 2)
b2 = np.random.randn(2)

np.savez(
    "checkpoint.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

# Loading:

checkpoint = np.load("checkpoint.npz")

W1 = checkpoint["W1"]
b1 = checkpoint["b1"]
W2 = checkpoint["W2"]
b2 = checkpoint["b2"]



#Saving Metadata
#We also want to know:
#Which epoch?
#Best validation loss?
#Learning rate?

# Example:

checkpoint = {
    "epoch": 17,
    "best_loss": 0.214,
    "learning_rate": 0.001
}
"""
For our framework, we'll save metadata separately (for example as JSON) alongside the NumPy arrays.
Project Structure
Our final project will contain:
checkpoints/
best_model.npz
last_model.npz
training_history.json
"""
