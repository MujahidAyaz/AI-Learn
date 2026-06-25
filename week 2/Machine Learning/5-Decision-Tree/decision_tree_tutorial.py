# ============================================================
#  DECISION TREE — Complete Beginner Tutorial
#  Problem: Predict whether a patient has diabetes
#  Dataset: Pima Indians Diabetes Dataset (768 patients)
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — IMPORT TOOLS
# ────────────────────────────────────────────────────────────
# DecisionTreeClassifier → the model
# plot_tree / export_text → to visualize and read what the model learned
# No StandardScaler needed — trees don't need feature scaling!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree         import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics      import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA & SPLIT
# ────────────────────────────────────────────────────────────
# Same diabetes dataset as logistic regression.
# Key difference: NO scaling step — trees ask "above or below X?"
# so the actual size of numbers doesn't matter to them.

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)

X = df.drop("Outcome", axis=1)   # features (clues)
y = df["Outcome"]                  # label (0=no diabetes, 1=diabetes)

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 55)
print("STEP 2 — DATA LOADED")
print("=" * 55)
print(f"Training patients : {len(X_train)}")
print(f"Testing  patients : {len(X_test)}")
print(f"Features          : {feature_names}")


# ────────────────────────────────────────────────────────────
# STEP 3 — OVERFIT TREE (no limits — to show the problem)
# ────────────────────────────────────────────────────────────
# A decision tree with no limits will grow until it perfectly
# classifies every single training patient.
# It memorizes — 100% train accuracy, bad test accuracy.
# We do this on purpose to SEE the overfitting problem.

tree_overfit = DecisionTreeClassifier(random_state=42)   # no limits!
tree_overfit.fit(X_train, y_train)

train_acc_overfit = accuracy_score(y_train, tree_overfit.predict(X_train))
test_acc_overfit  = accuracy_score(y_test,  tree_overfit.predict(X_test))

print()
print("=" * 55)
print("STEP 3 — OVERFIT TREE (no pruning)")
print("=" * 55)
print(f"Tree depth        : {tree_overfit.get_depth()}")
print(f"Training accuracy : {train_acc_overfit * 100:.1f}%   ← memorized!")
print(f"Test accuracy     : {test_acc_overfit  * 100:.1f}%   ← fails on new data")
print("Problem: gap between train and test is too large")


# ────────────────────────────────────────────────────────────
# STEP 4 — PRUNED TREE (fix overfitting)
# ────────────────────────────────────────────────────────────
# We limit how deep the tree can grow using:
#
#   max_depth=4          → only 4 questions deep maximum
#   min_samples_split=10 → only split if group has 10+ patients
#   min_samples_leaf=5   → each final leaf must have 5+ patients
#   criterion="gini"     → use Gini Impurity to pick best splits
#
# Gini Impurity measures how "mixed" a group is:
#   0.0 = perfectly pure  (all same class → great split)
#   0.5 = perfectly mixed (50/50 → useless split)
# The tree always picks the split that gives the lowest Gini.

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion="gini",
    random_state=42
)

tree.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree.predict(X_train))
test_acc  = accuracy_score(y_test,  tree.predict(X_test))

print()
print("=" * 55)
print("STEP 4 — PRUNED TREE (max_depth=4)")
print("=" * 55)
print(f"Tree depth        : {tree.get_depth()}")
print(f"Training accuracy : {train_acc * 100:.1f}%")
print(f"Test accuracy     : {test_acc  * 100:.1f}%")
print("Small gap = model generalizes well to new patients")


# ────────────────────────────────────────────────────────────
# STEP 5 — READ THE TREE (what did it actually learn?)
# ────────────────────────────────────────────────────────────
# This is a major advantage of decision trees over logistic
# regression — you can literally read the model's rules.
# A doctor can look at this and understand every decision.

print()
print("=" * 55)
print("STEP 5 — DECISION RULES (read the model's thinking)")
print("=" * 55)
tree_rules = export_text(tree, feature_names=feature_names)
print(tree_rules)

# The first question is almost always Glucose — the strongest
# predictor of diabetes in this dataset.


# ────────────────────────────────────────────────────────────
# STEP 6 — VISUALIZE THE TREE AS A DIAGRAM
# ────────────────────────────────────────────────────────────
# We draw the tree as a proper visual diagram.
# Each box (node) shows:
#   - The question being asked
#   - Gini score (how mixed the group is)
#   - Number of patients in this group
#   - Current majority class
#
# Orange boxes → majority are diabetic
# Blue boxes   → majority are non-diabetic
# Darker color → purer group (more confident answer)

fig, ax = plt.subplots(figsize=(20, 8))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=["No Diabetes", "Diabetes"],
    filled=True,       # color by majority class
    rounded=True,      # rounded box corners
    fontsize=10,
    ax=ax
)

ax.set_title("Decision Tree — Diabetes Prediction (max_depth=4)",
             fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("decision_tree_diagram.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Tree diagram saved to: decision_tree_diagram.png")


# ────────────────────────────────────────────────────────────
# STEP 7 — EVALUATE ON TEST SET
# ────────────────────────────────────────────────────────────
# Final exam. Show the model 154 patients it never saw.
# Then check feature importance — which clues did it use most?

y_pred  = tree.predict(X_test)
y_proba = tree.predict_proba(X_test)[:, 1]

print()
print("=" * 55)
print("STEP 7 — TEST SET EVALUATION")
print("=" * 55)
print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.1f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
print()
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes", "Diabetes"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Rows=Actual, Cols=Predicted):")
print(f"  True  Negatives (said No,  actually No)  : {cm[0][0]}")
print(f"  False Positives (said Yes, actually No)  : {cm[0][1]}")
print(f"  False Negatives (said No,  actually Yes) : {cm[1][0]}  ← missed cases!")
print(f"  True  Positives (said Yes, actually Yes) : {cm[1][1]}")

# Feature importance — how much did each feature contribute?
print()
print("Feature Importance (which clues the tree relied on):")
importances = tree.feature_importances_
for name, imp in sorted(zip(feature_names, importances),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {name:<30} {imp:.3f}  {bar}")

# Glucose is almost always the most important feature.
# SkinThickness often has 0 importance — the tree never used it.


# ────────────────────────────────────────────────────────────
# STEP 8 — PREDICT A BRAND NEW PATIENT
# ────────────────────────────────────────────────────────────
# No scaling needed — just plug in raw values.
# You can also trace the exact path through the tree manually.

print()
print("=" * 55)
print("STEP 8 — PREDICT A NEW PATIENT")
print("=" * 55)

new_patient = pd.DataFrame(
    [[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
    columns=feature_names
)

prediction  = tree.predict(new_patient)[0]
probability = tree.predict_proba(new_patient)[0][1]

print("Patient info:")
for col, val in zip(feature_names, new_patient.values[0]):
    print(f"  {col}: {val}")

print()
print(f"Prediction  : {'⚠ Diabetes Likely' if prediction == 1 else '✓ No Diabetes'}")
print(f"Confidence  : {probability * 100:.1f}% probability of diabetes")
print()
print("Tree's reasoning (traceable!):")
print("  Q1: Is Glucose > 127.5?  138 > 127.5  → YES")
print("  Q2: Is BMI > 29.95?      33.6 > 29.95 → YES")
print("  → Final answer: Diabetes")


# ────────────────────────────────────────────────────────────
# STEP 9 — VISUALIZATIONS SUMMARY
# ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Decision Tree — Diabetes Prediction", fontsize=14, fontweight="bold")

# Chart 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#52d48a", lw=2, label=f"AUC = {auc:.2f}")
axes[0].plot([0,1],[0,1],"--", color="#555")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Chart 2: Feature Importance
sorted_idx = np.argsort(importances)
axes[1].barh(np.array(feature_names)[sorted_idx],
             importances[sorted_idx], color="#f5a623")
axes[1].set_title("Feature Importance")
axes[1].set_xlabel("Importance Score")

# Chart 3: Confusion Matrix
axes[2].imshow(cm, cmap="YlOrBr")
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
plt.savefig("decision_tree_results.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Summary charts saved to: decision_tree_results.png")

# ============================================================
#  CLEAN VERSION BELOW — full code without comments
# ============================================================
