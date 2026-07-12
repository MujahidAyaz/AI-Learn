import numpy as np

np.random.seed(42)

class DataLoader:

    """Simple DataLoader similar to PyTorch."""

    def __init__(self, x,y, batch_size=4, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
       
        if self.shuffle:
            indices = np.random.permutation(len(self.x))
            x = self.x[indices]
            y = self.y[indices]
        else:
            x = self.x
            y = self.y
        for start in range(0, len(x), self.batch_size):
            end = start + self.batch_size
            yield x[start:end], y[start:end]
# Dataset
x=np.arange(1,21, dtype=np.float32)
y = 2 * x

loader = DataLoader(x,y, batch_size=4, shuffle=True)
for batch_number, (x_batch, y_batch) in enumerate(loader, start=1):
    print(f"Batch {batch_number}")
    print("Inputs :", x_batch)
    print("Targets:", y_batch)
    print("-" * 40)