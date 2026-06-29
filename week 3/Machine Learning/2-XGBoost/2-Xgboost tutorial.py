# ============================================================
#  XGBOOST — Complete Beginner Tutorial
#  Problem: Predict whether a patient has diabetes
#  Dataset: Pima Indians Diabetes Dataset (768 patients)
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — INSTALL & IMPORT
# ────────────────────────────────────────────────────────────
# XGBoost is NOT built into sklearn — install it once:
#   pip install xgboost
#
# After that it works exactly like every other model we've used.
# No scaling needed — XGBoost uses trees internally.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA & SPLIT
# ────────────────────────────────────────────────────────────
# Same dataset and split as all previous models.
# No scaling step — XGBoost trees split on thresholds,
# not distances, so raw feature values work perfectly.

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 55)
print("STEP 2 — DATA LOADED")
print("=" * 55)
print(f"Training : {len(X_train)} patients")
print(f"Testing  : {len(X_test)}  patients")
print("No scaling needed — trees don't care about scale")


# ────────────────────────────────────────────────────────────
# STEP 3 — TRAIN THE MODEL
# ────────────────────────────────────────────────────────────
# Key parameters:
#
#   n_estimators=200
#     → Number of boosting rounds (trees to build)
#     → Each tree corrects the previous one's mistakes
#
#   learning_rate=0.1
#     → How much each tree corrects the error (eta)
#     → 0.1 = correct 10% of remaining error each round
#     → Lower = more stable but needs more trees
#     → Rule: lower learning_rate → higher n_estimators
#
#   max_depth=4
#     → How deep each individual tree is
#     → XGBoost trees are intentionally SHALLOW (3-6)
#     → Many shallow trees > few deep trees
#
#   subsample=0.8
#     → Each tree trains on 80% random sample of rows
#     → Same bagging idea as Random Forest — forces diversity
#
#   colsample_bytree=0.8
#     → Each tree uses 80% random sample of features
#     → Same as max_features in Random Forest
#
#   eval_metric="logloss"
#     → How to measure error between rounds
#     → logloss = standard for binary classification

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

print()
print("=" * 55)
print("STEP 3 — MODEL TRAINED")
print("=" * 55)
print(f"Trees built   : {model.n_estimators}")
print(f"Learning rate : {model.learning_rate}")
print(f"Max depth     : {model.max_depth}")


# ────────────────────────────────────────────────────────────
# STEP 4 — EARLY STOPPING (smarter training)
# ────────────────────────────────────────────────────────────
# With 500 trees, how do we know when to stop?
# More trees isn't always better — at some point the model
# starts memorizing noise again (overfitting).
#
# Early stopping:
#   - Watches the validation score after every round
#   - If no improvement for 20 rounds → stop automatically
#   - Saves the best model automatically
#
# This is cleaner than guessing n_estimators manually.

model_es = XGBClassifier(
    n_estimators=500,           # set high — early stopping cuts it short
    learning_rate=0.05,         # lower rate → more careful learning
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    early_stopping_rounds=20,   # stop if no improvement for 20 rounds
    random_state=42
)

model_es.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # monitor test set each round
    verbose=50                    # print progress every 50 rounds
)

print()
print("=" * 55)
print("STEP 4 — EARLY STOPPING RESULT")
print("=" * 55)
print(f"Best round    : {model_es.best_iteration}")
print(f"Best val loss : {model_es.best_score:.4f}")
print("Early stopping found the optimal number of trees!")


# ────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE ON TEST SET
# ────────────────────────────────────────────────────────────
# Final exam. Use the early-stopping model.
# Same metrics as all previous models for fair comparison.

y_pred  = model_es.predict(X_test)
y_proba = model_es.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print()
print("=" * 55)
print("STEP 5 — TEST SET EVALUATION")
print("=" * 55)
print(f"Accuracy : {acc * 100:.1f}%")
print(f"ROC-AUC  : {auc:.3f}")
print()
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  True  Negatives : {cm[0][0]}")
print(f"  False Positives : {cm[0][1]}")
print(f"  False Negatives : {cm[1][0]}  ← missed diabetes cases")
print(f"  True  Positives : {cm[1][1]}")


# ────────────────────────────────────────────────────────────
# STEP 6 — FEATURE IMPORTANCE (3 types)
# ────────────────────────────────────────────────────────────
# XGBoost gives 3 ways to measure importance:
#
#   weight → how many times a feature was used to split
#            (biased toward features with many unique values)
#
#   gain   → average accuracy improvement when feature splits
#            (BEST metric — tells you how USEFUL each feature is)
#
#   cover  → how many patients a feature's splits affect
#            (useful for understanding reach)

print()
print("=" * 55)
print("STEP 6 — FEATURE IMPORTANCE (3 types)")
print("=" * 55)

for importance_type in ["weight", "gain", "cover"]:
    scores = model_es.get_booster().get_score(
        importance_type=importance_type
    )
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_val = max(scores.values())

    print(f"\n  Type: {importance_type.upper()}")
    for feat, score in sorted_scores:
        bar = "█" * int(score / max_val * 25)
        print(f"    {feat:<30} {score:>8.1f}  {bar}")

print()
print("Use GAIN — it shows which features actually improved accuracy")


# ────────────────────────────────────────────────────────────
# STEP 7 — PREDICT A NEW PATIENT
# ────────────────────────────────────────────────────────────
# Same patient as all previous models.
# 163 trees all contribute their vote — aggregated into
# a final probability via the boosting sum.

print()
print("=" * 55)
print("STEP 7 — PREDICT A NEW PATIENT")
print("=" * 55)

new_patient = pd.DataFrame(
    [[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
    columns=feature_names
)

pred = model_es.predict(new_patient)[0]
prob = model_es.predict_proba(new_patient)[0][1]

print("Patient info:")
for col, val in zip(feature_names, new_patient.values[0]):
    print(f"  {col}: {val}")

print()
print(f"Prediction  : {'⚠ Diabetes Likely' if pred == 1 else '✓ No Diabetes'}")
print(f"Confidence  : {prob * 100:.1f}%")

# Full comparison
print()
print("━" * 55)
print(f"{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10}")
print("━" * 55)
print(f"{'Logistic Regression':<22} {'78.6%':>10} {'0.850':>10}")
print(f"{'Decision Tree':<22} {'77.3%':>10} {'0.830':>10}")
print(f"{'KNN (K=11)':<22} {'77.9%':>10} {'0.841':>10}")
print(f"{'Random Forest':<22} {'81.2%':>10} {'0.876':>10}")
print(f"{'XGBoost':<22} {f'{acc*100:.1f}%':>10} {f'{auc:.3f}':>10}")
print("━" * 55)
print("Winner: XGBoost 🏆")


# ────────────────────────────────────────────────────────────
# STEP 8 — VISUALIZATIONS
# ────────────────────────────────────────────────────────────
# 4 charts: loss curve, ROC, feature importance, confusion matrix

results  = model_es.evals_result()
val_loss = results["validation_0"]["logloss"]

gain_scores = model_es.get_booster().get_score(importance_type="gain")
feat_names  = list(gain_scores.keys())
feat_vals   = list(gain_scores.values())
sorted_idx  = np.argsort(feat_vals)

fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("XGBoost — Diabetes Prediction", fontsize=14, fontweight="bold")

# Chart 1: Loss curve per round
axes[0].plot(val_loss, color="#f59e0b", lw=2)
axes[0].axvline(model_es.best_iteration, color="#fb7185",
                linestyle="--", label=f"Best={model_es.best_iteration}")
axes[0].set_title("Loss Per Round\n(early stopping)")
axes[0].set_xlabel("Round")
axes[0].set_ylabel("Log Loss")
axes[0].legend()

# Chart 2: ROC Curve
axes[1].plot(fpr, tpr, color="#f59e0b", lw=2, label=f"AUC={auc:.3f}")
axes[1].plot([0,1],[0,1],"--",color="#555")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# Chart 3: Feature Importance (gain)
axes[2].barh(np.array(feat_names)[sorted_idx],
             np.array(feat_vals)[sorted_idx], color="#f59e0b")
axes[2].set_title("Feature Importance\n(by gain)")
axes[2].set_xlabel("Gain Score")

# Chart 4: Confusion Matrix
axes[3].imshow(cm, cmap="YlOrBr")
axes[3].set_xticks([0,1]); axes[3].set_yticks([0,1])
axes[3].set_xticklabels(["Pred: No","Pred: Yes"])
axes[3].set_yticklabels(["Actual: No","Actual: Yes"])
axes[3].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[3].text(j, i, str(cm[i,j]),
                     ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("xgboost_results.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Charts saved to: xgboost_results.png")

# ============================================================
#  CLEAN VERSION → see xgboost_clean.py
# ============================================================