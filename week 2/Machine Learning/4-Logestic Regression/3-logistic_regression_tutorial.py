# ============================================================
#  LOGISTIC REGRESSION — Complete Beginner Tutorial
#  Problem: Predict whether a patient has diabetes
#  Dataset: Pima Indians Diabetes Dataset (768 patients)
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — IMPORT TOOLS
# ────────────────────────────────────────────────────────────
# Before cooking, you grab your tools.
# These libraries are pre-built — no need to write from scratch.

import pandas as pd               # for loading and handling tables of data
import numpy as np                # for math and number operations
import matplotlib.pyplot as plt   # for drawing charts

from sklearn.model_selection import train_test_split   # to split data
from sklearn.preprocessing  import StandardScaler      # to scale features
from sklearn.linear_model   import LogisticRegression  # the model itself
from sklearn.metrics        import (
    accuracy_score,        # overall % correct
    classification_report, # precision, recall, f1
    confusion_matrix,      # table of correct vs wrong predictions
    roc_auc_score,         # how well model ranks predictions
    roc_curve              # for drawing the ROC chart
)


# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD THE DATA
# ────────────────────────────────────────────────────────────
# We load a real dataset of 768 patients.
# Each row = one patient.
# Columns (features / clues):
#   Pregnancies, Glucose, BloodPressure, SkinThickness,
#   Insulin, BMI, DiabetesPedigreeFunction, Age
# Last column (label / answer):
#   Outcome → 1 = has diabetes, 0 = does not have diabetes

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("=" * 50)
print("STEP 2 — DATA LOADED")
print("=" * 50)
print(f"Total patients (rows): {df.shape[0]}")
print(f"Total columns:         {df.shape[1]}")
print()
print("First 5 rows:")
print(df.head())
print()
print("How many have diabetes vs not:")
print(df["Outcome"].value_counts())
# 0 = no diabetes → 500 patients
# 1 = has diabetes → 268 patients


# ────────────────────────────────────────────────────────────
# STEP 3 — SEPARATE FEATURES (X) AND LABEL (y)
# ────────────────────────────────────────────────────────────
# X = the clues the model will learn from (everything except Outcome)
# y = the answer we want to predict (just the Outcome column)

X = df.drop("Outcome", axis=1)   # all columns except the last
y = df["Outcome"]                 # just the last column

print("=" * 50)
print("STEP 3 — FEATURES AND LABEL SEPARATED")
print("=" * 50)
print(f"Features (X) shape: {X.shape}")   # (768, 8)
print(f"Label    (y) shape: {y.shape}")   # (768,)


# ────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────
# We split data into two piles:
#   Training set (80%) → model studies this
#   Test set     (20%) → hidden until the final exam
#
# Why? If you test on the same data you trained on,
# the model just memorizes — that's useless in real life.
# The test set gives you an honest score.
#
# random_state=42 → makes the split reproducible (same split every run)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42      # so results don't change each run
)

print()
print("=" * 50)
print("STEP 4 — TRAIN / TEST SPLIT")
print("=" * 50)
print(f"Training patients: {len(X_train)}")   # ~614
print(f"Testing  patients: {len(X_test)}")    # ~154


# ────────────────────────────────────────────────────────────
# STEP 5 — SCALE THE FEATURES
# ────────────────────────────────────────────────────────────
# Problem: Age goes 20–80. Glucose goes 0–200.
# The model might think glucose matters more just because
# its numbers are bigger. That's unfair and wrong.
#
# StandardScaler rescales every feature so they're
# all centered around 0 with similar spread.
# Example: Glucose 148 → becomes ~0.86 after scaling
#
# IMPORTANT RULE:
#   fit_transform on TRAINING data only → learns the scale
#   transform only on TEST data         → applies same scale
#   Never fit on test data — that would be cheating!

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn + apply
X_test_scaled  = scaler.transform(X_test)         # just apply

print()
print("=" * 50)
print("STEP 5 — FEATURES SCALED")
print("=" * 50)
print("Before scaling — Glucose mean:", round(X_train["Glucose"].mean(), 2))
print("After  scaling — Glucose mean:", round(X_train_scaled[:, 1].mean(), 4))
# After scaling, mean ≈ 0.0 and std ≈ 1.0


# ────────────────────────────────────────────────────────────
# STEP 6 — TRAIN THE MODEL
# ────────────────────────────────────────────────────────────
# This is the "studying" phase.
# Under the hood, sklearn does all of this for you:
#
#   1. Model makes predictions on all training patients
#   2. Loss function (Cross-Entropy) measures how wrong it was
#      → gave high confidence but was wrong = big penalty
#      → gave low confidence and was wrong  = smaller penalty
#   3. Gradient = which direction to adjust weights to reduce loss
#   4. Weights updated slightly in that direction
#   5. Repeat for max_iter loops
#
# Regularization (controlled by C):
#   Stops the model from memorizing noise in training data.
#   C=1.0  → default, balanced
#   C=0.1  → stronger regularization (simpler model)
#   C=10.0 → weaker regularization (more complex model)

model = LogisticRegression(
    C=1.0,          # regularization strength (1/lambda)
    max_iter=1000,  # max number of learning loops
    random_state=42
)

model.fit(X_train_scaled, y_train)   # ← all the learning happens here

print()
print("=" * 50)
print("STEP 6 — MODEL TRAINED")
print("=" * 50)
print("Training complete!")
print()

# Show what the model learned — weights for each feature
feature_names = X.columns.tolist()
weights = model.coef_[0]

print("Feature weights (how much each clue matters):")
for name, weight in sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑ increases risk" if weight > 0 else "↓ decreases risk"
    print(f"  {name:<30} {weight:+.3f}   {direction}")

# Glucose usually has the highest weight → strongest predictor


# ────────────────────────────────────────────────────────────
# STEP 7 — EVALUATE ON TEST SET
# ────────────────────────────────────────────────────────────
# Final exam. Show the model 154 patients it has NEVER seen.
# Compare predictions to actual reality.
#
# Accuracy  = overall % correct (can be misleading on imbalanced data)
# Precision = when it predicts YES, how often is it right?
# Recall    = of all actual YES cases, how many did it catch?
# F1-Score  = balance between precision and recall
# ROC-AUC   = overall quality score (0.5 = random, 1.0 = perfect)

y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]   # probability of diabetes

print()
print("=" * 50)
print("STEP 7 — TEST SET EVALUATION")
print("=" * 50)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy:  {acc * 100:.1f}%")
print(f"ROC-AUC:   {auc:.3f}")
print()
print("Detailed Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

print("Confusion Matrix:")
print("(Rows = Actual, Columns = Predicted)")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Pred: No   Pred: Yes")
print(f"Actual: No        {cm[0][0]:>4}        {cm[0][1]:>4}")
print(f"Actual: Yes       {cm[1][0]:>4}        {cm[1][1]:>4}")
# Top-left  = True Negatives  (correctly said NO)
# Top-right = False Positives (wrongly said YES)
# Bot-left  = False Negatives (missed actual YES — dangerous in medicine!)
# Bot-right = True Positives  (correctly said YES)


# ────────────────────────────────────────────────────────────
# STEP 8 — PREDICT A BRAND NEW PATIENT
# ────────────────────────────────────────────────────────────
# This is the whole point.
# A real new patient walks in — model has never seen them.
# We plug their info in and get an instant prediction.

print()
print("=" * 50)
print("STEP 8 — PREDICT A NEW PATIENT")
print("=" * 50)

# New patient data:
# [Pregnancies, Glucose, BloodPressure, SkinThickness,
#  Insulin, BMI, DiabetesPedigreeFunction, Age]
new_patient = pd.DataFrame([[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
                            columns=feature_names)

# Scale using the SAME scaler (important — must use same scale as training)
new_patient_scaled = scaler.transform(new_patient)

prediction  = model.predict(new_patient_scaled)[0]
probability = model.predict_proba(new_patient_scaled)[0][1]

print("Patient info:")
for col, val in zip(feature_names, new_patient.values[0]):
    print(f"  {col}: {val}")

print()
print(f"Prediction:  {'⚠ Diabetes Likely'  if prediction == 1 else '✓ No Diabetes'}")
print(f"Confidence:  {probability * 100:.1f}% probability of diabetes")


# ────────────────────────────────────────────────────────────
# STEP 9 — VISUALIZATIONS
# ────────────────────────────────────────────────────────────
# Three charts to understand what happened:
#   1. ROC Curve       → overall model quality
#   2. Feature weights → which clues matter most
#   3. Confusion matrix heatmap

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Logistic Regression — Diabetes Prediction", fontsize=14, fontweight="bold")

# ── Chart 1: ROC Curve ──────────────────────────────────────
# X-axis = False Positive Rate (how often we wrongly said YES)
# Y-axis = True Positive Rate / Recall (how often we caught YES)
# The closer the curve hugs the top-left corner, the better.
# A diagonal line = random guessing (AUC = 0.5)

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#3fb950", lw=2, label=f"ROC curve (AUC = {auc:.2f})")
axes[0].plot([0, 1], [0, 1], color="#555", linestyle="--", label="Random guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")
axes[0].set_facecolor("#0d1117")
axes[0].tick_params(colors="#aaa")

# ── Chart 2: Feature Importance ─────────────────────────────
# Shows which features the model learned to rely on most.
# Positive weight → increases probability of diabetes
# Negative weight → decreases probability of diabetes

sorted_idx = np.argsort(weights)
colors = ["#ff7b72" if w < 0 else "#3fb950" for w in weights[sorted_idx]]

axes[1].barh(np.array(feature_names)[sorted_idx], weights[sorted_idx], color=colors)
axes[1].axvline(0, color="#aaa", linewidth=0.8)
axes[1].set_xlabel("Weight (importance)")
axes[1].set_title("Feature Weights\n(green = raises risk, red = lowers risk)")
axes[1].set_facecolor("#0d1117")
axes[1].tick_params(colors="#aaa")

# ── Chart 3: Confusion Matrix ────────────────────────────────
# Visual table of correct vs wrong predictions

im = axes[2].imshow(cm, interpolation="nearest", cmap="Greens")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Pred: No", "Pred: Yes"])
axes[2].set_yticklabels(["Actual: No", "Actual: Yes"])
axes[2].set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

axes[2].set_facecolor("#0d1117")
axes[2].tick_params(colors="#aaa")

plt.tight_layout()
plt.savefig("logistic_regression_results.png", dpi=150, bbox_inches="tight",
            facecolor="#161b22")
plt.show()
print()
print("Charts saved to: logistic_regression_results.png")


# ============================================================
#  COMPLETE CODE IN ONE GO (no comments — clean version)
#  Copy everything below this line for a clean script.
# ============================================================
