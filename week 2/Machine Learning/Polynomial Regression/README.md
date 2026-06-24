# 📈 Polynomial Regression From Scratch

> **Built with NumPy only** — no sklearn for core logic. Every formula is hand-coded and explained.

---

## 🧠 What You Learn

| Concept | Where |
|---|---|
| What polynomial features are | `polynomial_features()` |
| Normal Equation (closed-form solution) | `fit()` |
| MSE & R² metrics | `mean_squared_error()`, `r_squared()` |
| Underfitting (degree 1) | `visualize_all_degrees()` |
| Good Fit (degree 2-3) | `visualize_all_degrees()` |
| Overfitting (degree 10+) | `train_test_overfit_demo()` |
| Train/Test split by hand | `train_test_overfit_demo()` |

---

## 🔑 Core Concepts

### Polynomial Features
```
x = 5, degree = 3
→ Feature vector: [1, 5, 25, 125]
                   ↑   ↑   ↑    ↑
                  w0  w1  w2   w3
```

### Normal Equation
```
W = (XᵀX)⁻¹ Xᵀy
```
No gradient descent. Pure matrix algebra. Exact solution!

---

## 📊 Underfitting vs Overfitting

```
Degree 1  → UNDERFIT  → straight line on a curved dataset (too simple)
Degree 2  → GOOD FIT  → captures the true curve
Degree 3  → GOOD FIT  → still great
Degree 10 → OVERFIT   → wiggly, memorizes noise, fails on new data
```

**The golden rule:**
- Underfit → High error on BOTH train and test
- Good fit  → Low error on BOTH train and test  ✅
- Overfit   → Low train error, HIGH test error

---

## 🚀 How to Run

```bash
pip install numpy matplotlib
python polynomial_regression.py
```

Output:
- Console table with MSE and R² for each degree
- Train vs Test error comparison (proof of overfitting)
- `polynomial_regression_plot.png` — visual of all 4 degrees

---

## 📁 Project Structure

```
polynomial_regression_project/
├── polynomial_regression.py   ← main file (fully commented)
└── README.md
```

---

## 📐 Math Reference

| Formula | Meaning |
|---|---|
| `y = w0 + w1x + w2x²` | Degree-2 polynomial |
| `W = (XᵀX)⁻¹Xᵀy` | Normal Equation |
| `MSE = mean((y - ŷ)²)` | Mean Squared Error |
| `R² = 1 - SS_res/SS_tot` | Coefficient of Determination |

---

*Built as a learning project — no sklearn, no shortcuts, just math and NumPy.*
