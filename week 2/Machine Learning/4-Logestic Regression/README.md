# Logistic Regression — Diabetes Prediction

A beginner-friendly implementation of Logistic Regression to predict whether a patient has diabetes, using the Pima Indians Diabetes Dataset.

## What this project covers

- Loading a real-world dataset
- Train/Test split
- Feature scaling
- Training a Logistic Regression model
- Evaluating with Accuracy, Precision, Recall, F1, ROC-AUC
- Predicting on a new patient
- Visualizing results (ROC Curve, Feature Weights, Confusion Matrix)

## Files

| File | Description |
|---|---|
| `logistic_regression_tutorial.py` | Step-by-step code with detailed plain-English comments |
| `logistic_regression_clean.py` | The complete code in one clean script (no comments) |

## How to run

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib scikit-learn
```

**2. Run the tutorial (with explanations)**
```bash
python logistic_regression_tutorial.py
```

**3. Or run the clean version**
```bash
python logistic_regression_clean.py
```

## Dataset

[Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv) — 768 patients, 8 features, binary outcome (diabetes yes/no).

## Results

| Metric | Score |
|---|---|
| Accuracy | ~78.6% |
| ROC-AUC | ~0.85 |

## Key concepts used

| Concept | What it does |
|---|---|
| **Train/Test Split** | 80% to learn, 20% for honest evaluation |
| **StandardScaler** | Puts all features on the same scale |
| **Loss Function** | Cross-entropy — measures how wrong each prediction is |
| **Gradient Descent** | Adjusts weights to reduce loss over 1000 iterations |
| **Regularization (C)** | Prevents model from memorizing noise |
| **Confusion Matrix** | Shows correct vs wrong predictions broken down by class |
