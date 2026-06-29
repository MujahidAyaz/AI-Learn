# ============================================================
#  K-NEAREST NEIGHBORS (KNN) — Complete Beginner Tutorial
#  Problem: Predict whether a patient has diabetes
#  Dataset: Pima Indians Diabetes Dataset (768 patients)
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — IMPORT TOOLS
# ────────────────────────────────────────────────────────────
# KNeighborsClassifier lives in sklearn.neighbors
# StandardScaler is BACK — KNN measures distances so scaling
# is absolutely critical (unlike decision tree / random forest)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA, SPLIT & SCALE
# ────────────────────────────────────────────────────────────
# Same dataset, same split as all previous models.
# Key difference from tree models: we MUST scale.
#
# Why? KNN measures straight-line distance between patients.
# Glucose ranges 0-200, Age ranges 20-80.
# Without scaling, Glucose dominates every distance calculation
# just because its numbers are bigger — completely unfair.
# After scaling both features are on the same playing field.

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale — fit on training data only, apply same scale to test
# Never fit on test data — that would be cheating!
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("=" * 55)
print("STEP 2 — DATA LOADED AND SCALED")
print("=" * 55)
print(f"Training patients : {len(X_train)}")
print(f"Testing  patients : {len(X_test)}")
print("Features scaled — ready for distance calculations")


# ────────────────────────────────────────────────────────────
# STEP 3 — FIND THE BEST K
# ────────────────────────────────────────────────────────────
# K = how many neighbors to look at when predicting.
#
# K=1  → copies the single nearest neighbor → overfitting
#         one weird patient throws off everything
# K=30 → looks at 30 neighbors → underfitting
#         too broad, ignores local patterns
#
# Solution: test every K from 1 to 30, pick whichever scores best.
# We use cross-validation (cv=5) for each K so the score is reliable.

print()
print("=" * 55)
print("STEP 3 — FINDING BEST K (testing K=1 to K=30)")
print("=" * 55)

k_values  = range(1, 31)
cv_scores = []

for k in k_values:
    knn   = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(
        knn, X_train_scaled, y_train,
        cv=5, scoring="roc_auc"
    ).mean()
    cv_scores.append(score)
    print(f"  K={k:<3}  ROC-AUC={score:.3f}")

best_k     = list(k_values)[int(np.argmax(cv_scores))]
best_score = max(cv_scores)

print()
print(f"Best K     : {best_k}")
print(f"Best Score : {best_score:.3f} ROC-AUC")

# Plot K vs score to visualize the sweet spot
plt.figure(figsize=(10, 4))
plt.plot(k_values, cv_scores, color="#60a5fa", lw=2, marker="o", markersize=4)
plt.axvline(best_k, color="#fb7185", linestyle="--", label=f"Best K={best_k}")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Cross-Val ROC-AUC")
plt.title("Finding the Best K for KNN")
plt.legend()
plt.tight_layout()
plt.savefig("knn_best_k.png", dpi=150, bbox_inches="tight")
plt.show()
print("K vs accuracy chart saved to: knn_best_k.png")


# ────────────────────────────────────────────────────────────
# STEP 4 — TRAIN THE FINAL MODEL
# ────────────────────────────────────────────────────────────
# "Training" KNN = just storing the scaled data in memory.
# No weights are learned. No trees are built.
# KNN is called a "lazy learner" — it does all the work
# at prediction time, not training time.
#
# metric="euclidean" → straight-line distance between patients
# weights="uniform"  → all K neighbors get equal vote
# weights="distance" → closer neighbors get more vote weight (alternative)

knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean",
    weights="uniform"
)

knn.fit(X_train_scaled, y_train)

print()
print("=" * 55)
print("STEP 4 — MODEL READY")
print("=" * 55)
print(f"Model          : KNN")
print(f"K              : {knn.n_neighbors} neighbors")
print(f"Distance metric: {knn.metric}")
print(f"Stored patients: {knn.n_samples_fit_}")
print("(KNN memorized all 614 training patients — no math, just storage)")


# ────────────────────────────────────────────────────────────
# STEP 5 — SEE HOW A SINGLE PREDICTION WORKS
# ────────────────────────────────────────────────────────────
# When predict() is called on a new patient, KNN:
#   1. Calculates distance from new patient to ALL 614 stored patients
#   2. Sorts them by distance (closest first)
#   3. Takes the top K (11 closest)
#   4. Counts how many have diabetes vs not
#   5. Majority wins → that's the prediction
#
# kneighbors() lets us inspect this process manually.

print()
print("=" * 55)
print("STEP 5 — TRACING A SINGLE PREDICTION")
print("=" * 55)

new_patient = pd.DataFrame(
    [[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
    columns=feature_names
)
new_scaled = scaler.transform(new_patient)

# Get distances and indices of K nearest neighbors
distances, indices = knn.kneighbors(new_scaled)

print(f"New patient features: {dict(zip(feature_names, new_patient.values[0]))}")
print()
print(f"The {best_k} nearest neighbors (sorted by distance):")
print(f"{'Rank':<6} {'Distance':>10} {'Has Diabetes?':>15}")
print("─" * 35)

for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    label = "YES ⚠" if y_train.iloc[idx] == 1 else "no  ✓"
    print(f"#{rank:<5} {dist:>10.3f} {label:>15}")

# Final vote
neighbor_labels = y_train.iloc[indices[0]]
yes_votes = int(neighbor_labels.sum())
no_votes  = best_k - yes_votes

print()
print(f"Vote result : {yes_votes} YES (Diabetes) vs {no_votes} NO (No Diabetes)")
print(f"Prediction  : {'⚠ Diabetes Likely' if yes_votes > no_votes else '✓ No Diabetes'}")
print(f"Confidence  : {yes_votes/best_k*100:.1f}% of neighbors have diabetes")


# ────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE ON TEST SET
# ────────────────────────────────────────────────────────────
# Final exam. 154 patients the model never saw.
# Same metrics as all previous models for fair comparison.

y_pred  = knn.predict(X_test_scaled)
y_proba = knn.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print()
print("=" * 55)
print("STEP 6 — TEST SET EVALUATION")
print("=" * 55)
print(f"Accuracy : {acc * 100:.1f}%")
print(f"ROC-AUC  : {auc:.3f}")
print()
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  True  Negatives : {cm[0][0]}  (correctly said No)")
print(f"  False Positives : {cm[0][1]}  (wrongly said Yes)")
print(f"  False Negatives : {cm[1][0]}  (missed diabetes ← important!)")
print(f"  True  Positives : {cm[1][1]}  (correctly said Yes)")


# ────────────────────────────────────────────────────────────
# STEP 7 — FULL MODEL COMPARISON
# ────────────────────────────────────────────────────────────
# All 4 models, same dataset, same split, honest comparison.

print()
print("=" * 55)
print("STEP 7 — ALL 4 MODELS COMPARED")
print("=" * 55)
print("━" * 65)
print(f"{'Model':<22} {'Accuracy':>9} {'ROC-AUC':>9} {'Scaling':>9} {'Training':>10}")
print("━" * 65)
print(f"{'Logistic Regression':<22} {'78.6%':>9} {'0.850':>9} {'Yes':>9} {'Yes':>10}")
print(f"{'Decision Tree':<22} {'77.3%':>9} {'0.830':>9} {'No':>9} {'Yes':>10}")
print(f"{'Random Forest':<22} {'81.2%':>9} {'0.876':>9} {'No':>9} {'Yes':>10}")
print(f"{'KNN (K='+str(best_k)+')':<22} {f'{acc*100:.1f}%':>9} {f'{auc:.3f}':>9} {'Yes':>9} {'No':>10}")
print("━" * 65)
print("Best accuracy    : Random Forest (81.2%) 🏆")
print("Most explainable : Decision Tree")
print("Simplest concept : KNN — find neighbors, copy majority")


# ────────────────────────────────────────────────────────────
# STEP 8 — VISUALIZATIONS
# ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("KNN — Diabetes Prediction", fontsize=14, fontweight="bold")

# Chart 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#60a5fa", lw=2, label=f"KNN K={best_k} (AUC={auc:.2f})")
axes[0].plot([0,1],[0,1],"--",color="#555")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Chart 2: K vs ROC-AUC (sweet spot chart)
axes[1].plot(list(k_values), cv_scores, color="#60a5fa", lw=2, marker="o", markersize=4)
axes[1].axvline(best_k, color="#fb7185", linestyle="--", label=f"Best K={best_k}")
axes[1].set_title("K vs ROC-AUC\n(finding the sweet spot)")
axes[1].set_xlabel("K (number of neighbors)")
axes[1].set_ylabel("Cross-Val ROC-AUC")
axes[1].legend()

# Chart 3: Confusion Matrix
axes[2].imshow(cm, cmap="Blues")
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
plt.savefig("knn_results.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Charts saved to: knn_results.png")

# ============================================================
#  CLEAN VERSION → see knn_clean.py
# ============================================================