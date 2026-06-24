import numpy as np
import matplotlib.pyplot as plt

# --------------------
# 1. DATASET
# --------------------
X = np.array([1, 2, 3, 4, 5])
y = np.array([30, 40, 50, 60, 70])

# --------------------
# 2. PARAMETERS (START GUESS)
# --------------------
theta0 = 0
theta1 = 0

# --------------------
# 3. HYPOTHESIS (PREDICTION FUNCTION)
# --------------------
def predict(X, theta0, theta1):
    return theta0 + theta1 * X

# --------------------
# 4. COST FUNCTION (MSE)
# --------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --------------------
# 5. GRADIENT DESCENT TRAINING LOOP
# --------------------
learning_rate = 0.01
epochs = 1000
n = len(X)

cost_history = []  # ← store cost at every epoch for plotting

for epoch in range(epochs):

    # Step 1: Predict
    y_pred = predict(X, theta0, theta1)

    # Step 2: Compute Cost
    cost = mse(y, y_pred)
    cost_history.append(cost)  # ← save it

    # Step 3: Compute Gradients
    d_theta0 = (-2/n) * np.sum(y - y_pred)
    d_theta1 = (-2/n) * np.sum(X * (y - y_pred))

    # Step 4: Update Parameters
    theta0 = theta0 - learning_rate * d_theta0
    theta1 = theta1 - learning_rate * d_theta1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost = {cost:.4f}, theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")

# --------------------
# 6. FINAL MODEL
# --------------------
print("\nFinal parameters:")
print("theta0:", round(theta0, 4))
print("theta1:", round(theta1, 4))

# --------------------
# 7. TEST PREDICTION
# --------------------
print("\nPrediction for 6 hours:")
print(round(theta0 + theta1 * 6, 4))

# --------------------
# 8. VISUALIZATION
# --------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Linear Regression via Gradient Descent", fontsize=15, fontweight='bold')

# --- Plot 1: Regression Line ---
ax1 = axes[0]

# original data points
ax1.scatter(X, y, color='blue', s=100, zorder=5, label='Actual Data')

# learned regression line
X_line = np.linspace(0, 6, 100)
y_line = theta0 + theta1 * X_line
ax1.plot(X_line, y_line, color='red', linewidth=2, label=f'y = {theta0:.1f} + {theta1:.1f}x')

# draw error lines (residuals)
y_pred_final = predict(X, theta0, theta1)
for i in range(len(X)):
    ax1.plot([X[i], X[i]], [y[i], y_pred_final[i]],
             color='gray', linestyle='--', linewidth=1)

ax1.set_title("Regression Line vs Actual Data", fontsize=12)
ax1.set_xlabel("X (Input)")
ax1.set_ylabel("y (Output)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 6.5)
ax1.set_ylim(0, 90)

# --- Plot 2: Cost over Epochs ---
ax2 = axes[1]

ax2.plot(range(epochs), cost_history, color='green', linewidth=2)
ax2.set_title("Cost Decreasing over Epochs", fontsize=12)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Cost")
ax2.grid(True, alpha=0.3)

# mark where cost became very small
ax2.axhline(y=min(cost_history), color='red', linestyle='--',
            label=f'Min Cost = {min(cost_history):.4f}')
ax2.legend()

plt.tight_layout()
plt.savefig("linear_regression_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as linear_regression_plot.png")
