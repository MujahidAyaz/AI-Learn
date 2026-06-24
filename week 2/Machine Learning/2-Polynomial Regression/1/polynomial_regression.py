import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================

data = pd.read_csv("data.csv")

x = data["StudyHours"].values
y = data["Marks"].values

# =========================
# Create Polynomial Features
# Degree = 2
# =========================

X = np.column_stack(
    (
        np.ones(len(x)),  # bias
        x,
        x**2
    )
)

# =========================
# Gradient Descent Setup
# =========================

m, n = X.shape

theta = np.zeros(n)

learning_rate = 0.001
epochs = 10000

# =========================
# Training
# =========================

for epoch in range(epochs):

    predictions = X @ theta

    error = predictions - y

    gradient = (2 / m) * (X.T @ error)

    theta = theta - learning_rate * gradient

    if epoch % 1000 == 0:
        cost = np.mean(error ** 2)
        print(f"Epoch {epoch}: Cost = {cost:.4f}")

# =========================
# Results
# =========================

print("\nLearned Parameters:")
print(theta)

# =========================
# Prediction Example
# =========================

hours = 12

hours_feature = np.array(
    [
        1,
        hours,
        hours**2
    ]
)

predicted_marks = hours_feature @ theta

print(f"\nPredicted Marks for {hours} study hours: {predicted_marks:.2f}")

# =========================
# Visualization
# =========================

x_curve = np.linspace(min(x), max(x), 100)

X_curve = np.column_stack(
    (
        np.ones(len(x_curve)),
        x_curve,
        x_curve**2
    )
)

y_curve = X_curve @ theta

plt.scatter(x, y, label="Actual Data")
plt.plot(x_curve, y_curve, label="Polynomial Regression")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Performance Predictor")

plt.legend()

plt.savefig("graph.png")

plt.show()