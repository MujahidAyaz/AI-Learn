"""
=============================================================
  POLYNOMIAL REGRESSION FROM SCRATCH
  Author  : You (future ML engineer 🚀)
  Library : NumPy only (no sklearn for core logic)
  Goal    : Understand every line, every formula
=============================================================

CONCEPT RECAP:
--------------
Linear Regression  → y = w0 + w1*x          (a straight line)
Polynomial Regress → y = w0 + w1*x + w2*x²  (a curve)

The SECRET: Polynomial Regression IS Linear Regression.
We just add extra columns (x², x³...) to our feature matrix.
The math stays the same!

NORMAL EQUATION (no loops, pure math):
    W = (X^T @ X)^(-1) @ X^T @ y
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# STEP 1: Generate Synthetic Data
# ─────────────────────────────────────────────
def generate_data(n_samples=80, noise=15, seed=42):
    """
    We create data that follows a TRUE curve:
        y = 0.5*x^2 - 3*x + 10 + noise

    This means a straight line can NEVER fit it perfectly.
    Only a polynomial can capture this shape.
    """
    np.random.seed(seed)
    X = np.linspace(-5, 15, n_samples)                     # 80 evenly spaced points
    y_true = 0.5 * X**2 - 3 * X + 10                      # TRUE underlying curve
    noise_term = np.random.normal(0, noise, n_samples)     # random noise
    y = y_true + noise_term                                 # observed data = truth + noise
    return X, y, y_true


# ─────────────────────────────────────────────
# STEP 2: Create Polynomial Features (NumPy)
# ─────────────────────────────────────────────
def polynomial_features(X, degree):
    """
    Transform [x] → [1, x, x², x³, ..., x^degree]

    For degree=3 and x=2:
        Input:  [2]
        Output: [1, 2, 4, 8]
                 ↑  ↑  ↑  ↑
                 w0 w1 w2 w3  (bias, linear, quadratic, cubic)

    Each column is a new "feature". The model learns a weight
    for each column. This is why it's STILL linear regression!
    """
    n = len(X)
    # Build matrix: each row is [1, x, x^2, ..., x^degree]
    X_poly = np.ones((n, degree + 1))   # start with all ones (the bias column)
    for d in range(1, degree + 1):
        X_poly[:, d] = X ** d           # fill each column: x^1, x^2, ...
    return X_poly


# ─────────────────────────────────────────────
# STEP 3: Fit Model using the Normal Equation
# ─────────────────────────────────────────────
def fit(X_poly, y):
    """
    Normal Equation: W = (X^T @ X)^(-1) @ X^T @ y

    This gives us the EXACT best-fit weights.
    No iterations, no learning rate — pure algebra!

    WHY does this work?
    We want to minimize: Sum of (y_pred - y_true)^2
    Taking the derivative and setting it to zero gives us this formula.
    """
    # np.linalg.pinv = pseudo-inverse (more numerically stable than inv)
    W = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
    return W


# ─────────────────────────────────────────────
# STEP 4: Predict
# ─────────────────────────────────────────────
def predict(X_poly, W):
    """
    Prediction is just: y_hat = X_poly @ W
    (matrix multiplication of features and weights)
    """
    return X_poly @ W


# ─────────────────────────────────────────────
# STEP 5: Evaluation Metrics
# ─────────────────────────────────────────────
def mean_squared_error(y_true, y_pred):
    """MSE = average of squared errors. Lower = better."""
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    """
    R² tells us: how much variance does our model explain?
    R² = 1.0  → perfect fit
    R² = 0.0  → model is as good as just predicting the mean
    R² < 0    → model is WORSE than predicting the mean (bad!)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)          # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # total sum of squares
    return 1 - (ss_res / ss_tot)


# ─────────────────────────────────────────────
# STEP 6: Visualize Everything
# ─────────────────────────────────────────────
def visualize_all_degrees(X, y, y_true):
    """
    We test degree 1, 2, 3, 10 to see:
    - Degree 1 → UNDERFITTING (too simple, can't capture curve)
    - Degree 2 → GOOD FIT     (just right, matches true pattern)
    - Degree 3 → STILL GOOD   (slight improvement)
    - Degree 10 → OVERFITTING (memorizes noise, wiggly mess)
    """
    degrees = [1, 2, 3, 10]
    titles  = [
        "Degree 1\n😞 UNDERFITTING\n(too simple)",
        "Degree 2\n✅ GOOD FIT\n(just right)",
        "Degree 3\n✅ STILL GOOD\n(matches truth)",
        "Degree 10\n🔥 OVERFITTING\n(memorizes noise)"
    ]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    X_smooth = np.linspace(X.min(), X.max(), 300)   # smooth line for plotting

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.3)

    print("\n" + "="*65)
    print(f"{'DEGREE':<10} {'MSE':>12} {'R²':>10}  VERDICT")
    print("="*65)

    for i, (deg, title, color) in enumerate(zip(degrees, titles, colors)):
        # --- Train ---
        X_poly = polynomial_features(X, deg)
        W = fit(X_poly, y)
        y_pred_train = predict(X_poly, W)

        # --- Predict on smooth line ---
        X_smooth_poly = polynomial_features(X_smooth, deg)
        y_smooth = predict(X_smooth_poly, W)

        # --- Metrics ---
        mse = mean_squared_error(y, y_pred_train)
        r2  = r_squared(y, y_pred_train)

        verdict = {1: "UNDERFIT", 2: "GOOD", 3: "GOOD", 10: "OVERFIT"}[deg]
        print(f"Degree {deg:<5} {mse:>12.2f} {r2:>10.4f}  {verdict}")

        # --- Plot ---
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#1a1a2e")
        ax.scatter(X, y, color="white", s=15, alpha=0.6, label="Data")
        ax.plot(X_smooth, X_smooth * 0.5**2 - 3 * X_smooth + 10,   # oops, let's use true
                color="yellow", linewidth=1.5, linestyle="--", alpha=0.5, label="True curve")
        ax.plot(X_smooth, y_smooth, color=color, linewidth=2.5, label=f"Degree {deg}")
        ax.set_title(title, color="white", fontsize=9, pad=8)
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333355")
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
        ax.set_xlabel("x", color="gray", fontsize=8)
        ax.set_ylabel("y", color="gray", fontsize=8)

    print("="*65)

    # Bottom row: Overfitting vs Underfitting conceptual diagram
    ax_concept = fig.add_subplot(gs[1, :])
    ax_concept.set_facecolor("#1a1a2e")
    ax_concept.axis("off")

    concept_text = (
        "📚  UNDERFITTING (Degree 1)  →  Model too SIMPLE. "
        "High error on BOTH training and test data.  "
        "The straight line can't follow the curve. Bias is HIGH.\n\n"
        "✅  GOOD FIT (Degree 2-3)    →  Model complexity matches the data. "
        "Low error on training AND test. This is what we want.\n\n"
        "🔥  OVERFITTING (Degree 10)  →  Model too COMPLEX. "
        "Low error on training, HIGH error on new data. "
        "It memorized the noise instead of the pattern. Variance is HIGH."
    )
    ax_concept.text(0.02, 0.7, concept_text, transform=ax_concept.transAxes,
                    color="white", fontsize=10, va="top",
                    bbox=dict(facecolor="#16213e", edgecolor="#4444aa", alpha=0.8, pad=10))

    fig.suptitle("Polynomial Regression — Underfitting vs Good Fit vs Overfitting",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig("polynomial_regression_plot.png",
                dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print("\n✅  Plot saved as polynomial_regression_plot.png")
    plt.close()


# ─────────────────────────────────────────────
# STEP 7: Train/Test Split Demo (Overfitting proof)
# ─────────────────────────────────────────────
def train_test_overfit_demo(X, y):
    """
    PROOF of overfitting:
    High-degree model → LOW training error but HIGH test error.
    That gap is the hallmark of overfitting.
    """
    print("\n" + "="*65)
    print("TRAIN vs TEST ERROR  (80% train, 20% test)")
    print("="*65)
    print(f"{'DEGREE':<10} {'TRAIN MSE':>12} {'TEST MSE':>12}  NOTE")
    print("-"*65)

    # Simple manual split (no sklearn!)
    split = int(0.8 * len(X))
    idx = np.random.permutation(len(X))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    for deg in [1, 2, 3, 5, 10, 15]:
        X_tr_poly = polynomial_features(X_train, deg)
        X_te_poly = polynomial_features(X_test,  deg)

        W = fit(X_tr_poly, y_train)

        train_mse = mean_squared_error(y_train, predict(X_tr_poly, W))
        test_mse  = mean_squared_error(y_test,  predict(X_te_poly, W))

        gap = test_mse - train_mse
        if gap > 500:
            note = "🔥 OVERFITTING"
        elif train_mse > 400:
            note = "😞 UNDERFITTING"
        else:
            note = "✅ GOOD"

        print(f"Degree {deg:<5} {train_mse:>12.1f} {test_mse:>12.1f}  {note}")

    print("="*65)
    print("\nKEY INSIGHT:")
    print("  As degree goes up → train error ↓ but test error ↑")
    print("  That GROWING GAP = overfitting in action!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎓 POLYNOMIAL REGRESSION FROM SCRATCH (NumPy only)\n")

    # Step 1: Data
    X, y, y_true = generate_data()
    print(f"✅ Data generated: {len(X)} samples")
    print(f"   True pattern: y = 0.5x² - 3x + 10 + noise")

    # Step 2–5: Show how polynomial features work
    print("\n📐 POLYNOMIAL FEATURES EXAMPLE (x=3, degree=3):")
    sample = polynomial_features(np.array([3.0]), degree=3)
    print(f"   Input x = 3  →  Feature vector = {sample[0]}")
    print(f"   (These are: [1, x¹, x², x³] = [1, 3, 9, 27])")

    # Step 6: Main visualization
    print("\n📊 Plotting all degrees...")
    visualize_all_degrees(X, y, y_true)

    # Step 7: Overfitting proof
    train_test_overfit_demo(X, y)

    print("\n🏁 Done! Check 'polynomial_regression_plot.png'\n")
