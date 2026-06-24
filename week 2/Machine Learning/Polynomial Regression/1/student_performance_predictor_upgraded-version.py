"""
=============================================================
  STUDENT PERFORMANCE PREDICTOR
  Polynomial Regression with Gradient Descent
  
  UPGRADES over original:
  1. Auto-generates data.csv if missing (no crash!)
  2. Feature Scaling (Normalization) → faster, stable training
  3. Train / Test Split → detect overfitting
  4. MSE + R² metrics printed
  5. Cost history plot → see if training is working
  6. Cleaner, more informative visualization
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# UPGRADE 1: Auto-generate data if file missing
# ─────────────────────────────────────────────
def load_or_create_data(filepath="data.csv"):
    if os.path.exists(filepath):
        print(f"✅ Loaded '{filepath}'")
        data = pd.read_csv(filepath)
    else:
        print(f"⚠️  '{filepath}' not found — generating sample data...")
        np.random.seed(42)
        hours = np.round(np.random.uniform(1, 14, 100), 1)
        marks = 2 * hours**2 - 5 * hours + 30 + np.random.normal(0, 8, 100)
        marks = np.clip(marks, 0, 100)
        data = pd.DataFrame({"StudyHours": hours, "Marks": marks})
        data.to_csv(filepath, index=False)
        print(f"✅ Generated and saved '{filepath}' with 100 samples")

    return data["StudyHours"].values, data["Marks"].values


# ─────────────────────────────────────────────
# UPGRADE 2: Feature Scaling (Normalization)
# ─────────────────────────────────────────────
def normalize(X_col):
    """
    WHY normalize?
    x² can be 100x larger than x. Gradient descent gets confused
    when features are on wildly different scales. Normalizing
    puts everything between 0 and 1 → faster, stable training.

    Formula: x_norm = (x - min) / (max - min)
    """
    col_min = X_col.min()
    col_max = X_col.max()
    return (X_col - col_min) / (col_max - col_min), col_min, col_max


# ─────────────────────────────────────────────
# UPGRADE 3: Train / Test Split (manual, no sklearn)
# ─────────────────────────────────────────────
def train_test_split(x, y, test_size=0.2, seed=42):
    """
    Shuffle and split data into train (80%) and test (20%).
    WHY? To check if our model generalizes to NEW data.
    """
    np.random.seed(seed)
    idx = np.random.permutation(len(x))
    split = int(len(x) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# ─────────────────────────────────────────────
# Build Polynomial Feature Matrix (same as original)
# ─────────────────────────────────────────────
def build_features(x, degree=2):
    """[1, x, x²]  → shape: (n_samples, degree+1)"""
    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


# ─────────────────────────────────────────────
# Gradient Descent (same logic, now returns history)
# ─────────────────────────────────────────────
def gradient_descent(X, y, learning_rate=0.01, epochs=10000):
    """
    HOW IT WORKS:
    1. Start with random weights (zeros)
    2. Make a prediction
    3. Calculate error
    4. Nudge weights in the direction that reduces error
    5. Repeat thousands of times

    Gradient = direction of steepest increase in error
    We go the OPPOSITE direction → that's why we subtract
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []                          # UPGRADE: track cost over time

    for epoch in range(epochs):
        predictions = X @ theta
        error       = predictions - y
        gradient    = (2 / m) * (X.T @ error)
        theta       = theta - learning_rate * gradient

        cost = np.mean(error ** 2)
        cost_history.append(cost)

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:>6}: MSE = {cost:.4f}")

    return theta, cost_history


# ─────────────────────────────────────────────
# UPGRADE 4: Evaluation Metrics
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred, label=""):
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"  {label:<10} MSE = {mse:>8.2f}  |  R² = {r2:.4f}")
    return mse, r2


# ─────────────────────────────────────────────
# UPGRADE 5: Full visualization (3 panels)
# ─────────────────────────────────────────────
def visualize(x_train, y_train, x_test, y_test, theta, cost_history, x_raw):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    # --- Panel 1: Regression Curve ---
    ax = axes[0]
    x_curve = np.linspace(x_raw.min(), x_raw.max(), 200)
    # normalize x_curve the same way training data was normalized
    x_curve_norm, _, _ = normalize(x_curve)
    X_curve = build_features(x_curve_norm)
    y_curve = X_curve @ theta

    ax.scatter(x_train, y_train, color="#3498db", s=40, alpha=0.7, label="Train data")
    ax.scatter(x_test,  y_test,  color="#e74c3c", s=40, alpha=0.7, label="Test data")
    ax.plot(x_curve, y_curve, color="#2ecc71", linewidth=2.5, label="Polynomial fit")
    ax.set_xlabel("Study Hours", color="white")
    ax.set_ylabel("Marks", color="white")
    ax.set_title("Polynomial Regression Fit", color="white", fontsize=11)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # --- Panel 2: Cost History ---
    ax = axes[1]
    ax.plot(cost_history, color="#f39c12", linewidth=1.5)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Cost", color="white")
    ax.set_title("Training Loss Curve\n(Should go DOWN)", color="white", fontsize=11)
    ax.set_yscale("log")   # log scale shows the drop more clearly

    # --- Panel 3: Residuals (Actual vs Predicted) ---
    ax = axes[2]
    x_all_norm, _, _ = normalize(np.concatenate([x_train, x_test]))
    X_all = build_features(x_all_norm)
    y_all_pred = X_all @ theta
    y_all_true = np.concatenate([y_train, y_test])
    residuals = y_all_true - y_all_pred

    ax.scatter(y_all_pred, residuals, color="#9b59b6", s=30, alpha=0.7)
    ax.axhline(0, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Marks", color="white")
    ax.set_ylabel("Residuals (Error)", color="white")
    ax.set_title("Residual Plot\n(Good model → scattered around 0)", color="white", fontsize=11)

    plt.suptitle("Student Performance Predictor — Full Analysis",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graph.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print("\n✅  graph.png saved!")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("   STUDENT PERFORMANCE PREDICTOR")
    print("="*55)

    # 1. Load data
    x_raw, y = load_or_create_data("data.csv")

    # 2. Train/Test split
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(x_raw, y)
    print(f"\n📊 Train: {len(x_train_raw)} samples | Test: {len(x_test_raw)} samples")

    # 3. Normalize features (UPGRADE: prevents gradient instability)
    x_train_norm, x_min, x_max = normalize(x_train_raw)
    x_test_norm = (x_test_raw - x_min) / (x_max - x_min)  # same scale as train!

    # 4. Build polynomial features
    X_train = build_features(x_train_norm, degree=2)
    X_test  = build_features(x_test_norm,  degree=2)

    # 5. Train with gradient descent
    print("\n🏋️  Training...\n")
    theta, cost_history = gradient_descent(X_train, y_train,
                                           learning_rate=0.01,
                                           epochs=10000)

    # 6. Evaluate
    print("\n📈 EVALUATION:")
    y_train_pred = X_train @ theta
    y_test_pred  = X_test  @ theta
    evaluate(y_train, y_train_pred, label="Train")
    evaluate(y_test,  y_test_pred,  label="Test ")

    # 7. Predict a new value
    print("\n🔮 PREDICTION:")
    hours = 12
    hours_norm = (hours - x_min) / (x_max - x_min)
    hours_feat = np.array([1, hours_norm, hours_norm**2])
    predicted = hours_feat @ theta
    print(f"  Study Hours = {hours}  →  Predicted Marks = {predicted:.1f}")

    # 8. Learned weights
    print(f"\n⚙️  Learned Parameters (theta):")
    print(f"  theta = {theta}")

    # 9. Visualize
    visualize(x_train_raw, y_train, x_test_raw, y_test,
              theta, cost_history, x_raw)

    print("\n" + "="*55)
    print("  Done! Check graph.png")
    print("="*55 + "\n")
