# ============================================================
#  RANDOM FOREST — Complete Beginner Tutorial
#  Problem: Predict whether a patient has diabetes
#  Dataset: Pima Indians Diabetes Dataset (768 patients)
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — IMPORT TOOLS
# ────────────────────────────────────────────────────────────
# RandomForestClassifier lives in sklearn.ensemble
# "ensemble" = combining many models into one stronger model
# No StandardScaler needed — trees don't care about feature scale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA & SPLIT
# ────────────────────────────────────────────────────────────
# Same dataset as logistic regression and decision tree.
# Same split so results are directly comparable.
# No scaling step — Random Forest inherits this from trees.

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
print(f"Training patients : {len(X_train)}")
print(f"Testing  patients : {len(X_test)}")


# ────────────────────────────────────────────────────────────
# STEP 3 — TRAIN THE RANDOM FOREST
# ────────────────────────────────────────────────────────────
# Key parameters:
#
#   n_estimators = 100
#     → Build 100 decision trees
#     → More trees = more stable but slower
#     → 100-500 is usually good
#
#   max_depth = 6
#     → Limit each tree's depth (same pruning idea as before)
#     → Prevents individual trees from overfitting
#
#   max_features = "sqrt"
#     → At each split, each tree only considers √8 ≈ 3 random features
#     → Forces diversity — trees can't all ask the same questions
#
#   oob_score = True
#     → Each tree trains on ~80% of data (random sample)
#     → The leftover 20% it never saw = "out-of-bag" data
#     → sklearn tests each tree on its own OOB data automatically
#     → Free accuracy estimate — no need to waste your test set on tuning!
#
#   n_jobs = -1
#     → Use all CPU cores to train trees in parallel (faster)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

print()
print("=" * 55)
print("STEP 3 — RANDOM FOREST TRAINED")
print("=" * 55)
print(f"Trees built  : {rf.n_estimators}")
print(f"OOB Score    : {rf.oob_score_ * 100:.1f}%")
print("OOB = free accuracy estimate using each tree's leftover data")


# ────────────────────────────────────────────────────────────
# STEP 4 — SEE THE VOTING IN ACTION
# ────────────────────────────────────────────────────────────
# When you call predict(), all 100 trees vote.
# The class with most votes wins.
# predict_proba() returns the fraction of trees that voted YES.
# Example: 0.68 means 68 out of 100 trees said "Diabetes"

one_patient = X_test.iloc[[0]]
votes = [tree.predict(one_patient)[0] for tree in rf.estimators_]
yes_votes = sum(votes)
no_votes  = 100 - yes_votes

print()
print("=" * 55)
print("STEP 4 — VOTING FOR ONE PATIENT")
print("=" * 55)
print(f"YES votes (Diabetes)    : {yes_votes}")
print(f"NO  votes (No Diabetes) : {no_votes}")
print(f"Final decision          : {'Diabetes' if yes_votes > 50 else 'No Diabetes'}")
print(f"Confidence              : {yes_votes}% of trees agreed")


# ────────────────────────────────────────────────────────────
# STEP 5 — EVALUATE ON TEST SET
# ────────────────────────────────────────────────────────────
# Final exam. 154 patients the model never saw.
# Same metrics as before — easy to compare with previous models.

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

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
# STEP 6 — FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────
# feature_importances_ is averaged across ALL 100 trees.
# More reliable than a single tree — it's 100 opinions averaged.
# Every feature gets some importance (unlike the single tree
# where SkinThickness = 0.000).

importances = rf.feature_importances_

print()
print("=" * 55)
print("STEP 6 — FEATURE IMPORTANCE (averaged over 100 trees)")
print("=" * 55)
for name, imp in sorted(zip(feature_names, importances),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {name:<30} {imp:.3f}  {bar}")


# ────────────────────────────────────────────────────────────
# STEP 7 — CROSS-VALIDATION
# ────────────────────────────────────────────────────────────
# One train/test split can be lucky or unlucky.
# Cross-validation (cv=5) splits the data 5 different ways,
# trains and tests 5 times, and averages the scores.
# Much more reliable measure of true model performance.
#
# How cv=5 works:
#   Split 1: train on folds 2,3,4,5 → test on fold 1
#   Split 2: train on folds 1,3,4,5 → test on fold 2
#   Split 3: train on folds 1,2,4,5 → test on fold 3
#   ... and so on
#   Final score = average of all 5 test scores

cv_scores = cross_val_score(
    rf, X, y, cv=5, scoring="roc_auc", n_jobs=-1
)

print()
print("=" * 55)
print("STEP 7 — CROSS-VALIDATION (5 folds)")
print("=" * 55)
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.3f}")
print()
print(f"  Mean : {cv_scores.mean():.3f}")
print(f"  Std  : {cv_scores.std():.3f}  ← low = consistent")
print("  Low std means model isn't just lucky on one split")


# ────────────────────────────────────────────────────────────
# STEP 8 — PREDICT A NEW PATIENT
# ────────────────────────────────────────────────────────────
# Same patient as logistic regression and decision tree.
# Now 100 trees vote instead of one model deciding.

print()
print("=" * 55)
print("STEP 8 — PREDICT A NEW PATIENT")
print("=" * 55)

new_patient = pd.DataFrame(
    [[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
    columns=feature_names
)

pred = rf.predict(new_patient)[0]
prob = rf.predict_proba(new_patient)[0][1]

print(f"Prediction  : {'⚠ Diabetes Likely' if pred == 1 else '✓ No Diabetes'}")
print(f"Confidence  : {prob * 100:.1f}% (vote of 100 trees)")

# Compare all three models
print()
print("━" * 52)
print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
print("━" * 52)
print(f"{'Logistic Regression':<25} {'78.6%':>10} {'0.850':>10}")
print(f"{'Decision Tree':<25} {'77.3%':>10} {'0.830':>10}")
print(f"{'Random Forest':<25} {f'{acc*100:.1f}%':>10} {f'{auc:.3f}':>10}")
print("━" * 52)
print("Winner: Random Forest 🏆")


# ────────────────────────────────────────────────────────────
# STEP 9 — VISUALIZATIONS
# ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Random Forest — Diabetes Prediction", fontsize=14, fontweight="bold")

# Chart 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#4ade80", lw=2, label=f"Random Forest (AUC={auc:.2f})")
axes[0].plot([0,1],[0,1],"--",color="#555")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Chart 2: Feature Importance
sorted_idx = np.argsort(importances)
axes[1].barh(np.array(feature_names)[sorted_idx],
             importances[sorted_idx], color="#4ade80")
axes[1].set_title("Feature Importance\n(averaged over 100 trees)")
axes[1].set_xlabel("Importance Score")

# Chart 3: Confusion Matrix
axes[2].imshow(cm, cmap="Greens")
axes[2].set_xticks([0,1]); axes[2].set_yticks([0,1])
axes[2].set_xticklabels(["Pred: No","Pred: Yes"])
axes[2].set_yticklabels(["Actual: No","Actual: Yes"])
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i,j]),
                     ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("random_forest_results.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Charts saved to: random_forest_results.png")

# ============================================================
#  CLEAN VERSION BELOW — see random_forest_clean.py
# ============================================================
