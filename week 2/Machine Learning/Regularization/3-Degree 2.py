import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def polynomial_features(x, degree):
    return np.column_stack([x**d for d in range(degree + 1)])


def train(X, y, lr=0.00001, epochs=50000):
    m, n = X.shape
    theta = np.zeros(n)
    for epoch in range(epochs):
        error = X @ theta - y
        theta -= lr * (2 / m) * (X.T @ error)
    return theta


# Load data
df = pd.read_csv("Regularization_data.csv")
x = df["Hours"].values
y = df["Marks"].values

# Feature scaling
x = (x - x.mean()) / x.std()

degrees = [2, 5, 10]
x_curve = np.linspace(x.min(), x.max(), 500)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Actual Data", color="black", zorder=5)

for degree in degrees:
    X = polynomial_features(x, degree)
    theta = train(X, y)
    y_curve = polynomial_features(x_curve, degree) @ theta
    plt.plot(x_curve, y_curve, linewidth=2, label=f"Degree {degree}")

plt.xlabel("Scaled Hours")
plt.ylabel("Marks")
plt.title("Polynomial Regression — Degree Comparison")
plt.legend()
plt.grid(True)
plt.show()