import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================
# Feature Engineering
# =====================================

def polynomial_features(x, degree):
    return np.column_stack(
        [x ** d for d in range(degree + 1)]
    )


# =====================================
# Training
# =====================================

def train(
    X,
    y,
    lr=0.00001,
    epochs=50000,
    lambda_=0
):

    m, n = X.shape

    theta = np.zeros(n)

    for epoch in range(epochs):

        predictions = X @ theta

        error = predictions - y

        gradient = (
            (2 / m) * (X.T @ error)
            +
            2 * lambda_ * theta
        )

        theta -= lr * gradient

    return theta


# =====================================
# Load Data
# =====================================

df = pd.read_csv(
    "Regularization_data.csv"
)

x = df["Hours"].values
y = df["Marks"].values


# =====================================
# Feature Scaling
# =====================================

x = (
    x - x.mean()
) / x.std()


# =====================================
# Degree 10 Features
# =====================================

degree = 10

X = polynomial_features(
    x,
    degree
)


# =====================================
# Train Models
# =====================================

theta_no_ridge = train(
    X,
    y,
    lambda_=0
)

theta_ridge = train(
    X,
    y,
    lambda_=1
)


# =====================================
# Smooth Curve
# =====================================

x_curve = np.linspace(
    x.min(),
    x.max(),
    1000
)

X_curve = polynomial_features(
    x_curve,
    degree
)

y_no_ridge = (
    X_curve @ theta_no_ridge
)

y_ridge = (
    X_curve @ theta_ridge
)


# =====================================
# MSE
# =====================================

mse_no_ridge = np.mean(
    (X @ theta_no_ridge - y) ** 2
)

mse_ridge = np.mean(
    (X @ theta_ridge - y) ** 2
)

print(
    f"No Ridge MSE: {mse_no_ridge:.2f}"
)

print(
    f"Ridge MSE: {mse_ridge:.2f}"
)


# =====================================
# Figure Layout
# =====================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7)
)


# =====================================
# Chart 1
# Model Comparison
# =====================================

axes[0].scatter(
    x,
    y,
    s=70,
    alpha=0.8,
    label="Actual Data"
)

axes[0].plot(
    x_curve,
    y_no_ridge,
    linewidth=3,
    label="Degree 10 (No Ridge)"
)

axes[0].plot(
    x_curve,
    y_ridge,
    linewidth=3,
    label="Degree 10 + Ridge"
)

axes[0].set_title(
    "Polynomial Regression: Overfitting vs Ridge",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_xlabel(
    "Scaled Hours"
)

axes[0].set_ylabel(
    "Marks"
)

axes[0].grid(
    alpha=0.3
)

axes[0].legend()


# =====================================
# Chart 2
# Coefficient Comparison
# =====================================

indices = np.arange(
    len(theta_no_ridge)
)

width = 0.35

axes[1].bar(
    indices - width/2,
    theta_no_ridge,
    width,
    label="No Ridge"
)

axes[1].bar(
    indices + width/2,
    theta_ridge,
    width,
    label="Ridge"
)

axes[1].set_title(
    "Coefficient Shrinkage (Theta Values)",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Coefficient Index"
)

axes[1].set_ylabel(
    "Coefficient Value"
)

axes[1].legend()

axes[1].grid(
    alpha=0.3
)


# =====================================
# Main Title
# =====================================

fig.suptitle(
    "Ridge Regularization Analysis",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()