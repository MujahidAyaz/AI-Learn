"""
===============================================================
Project: Principal Component Analysis (PCA)

Dataset:
Breast Cancer Wisconsin Dataset
(Built into Scikit-Learn)

Algorithm:
Principal Component Analysis (PCA)

Classifier:
Logistic Regression

Author:
Mujahid_Ayaz

Description:
This project demonstrates how PCA reduces the number of
features while preserving as much information as possible.
The project also compares model performance before and
after applying PCA.

===============================================================
"""

# ============================================================
# STEP 1 - Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset
from sklearn.datasets import load_breast_cancer

# Split Dataset
from sklearn.model_selection import train_test_split

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# PCA
from sklearn.decomposition import PCA

# Classification Model
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# STEP 2 - Load Dataset
# ============================================================

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

print("=" * 60)
print("First Five Rows")
print(df.head())

# ============================================================
# STEP 3 - Explore Dataset
# ============================================================

print("=" * 60)
print("Dataset Shape")
print(df.shape)

print("=" * 60)
print("Dataset Information")
print(df.info())

print("=" * 60)
print("Missing Values")
print(df.isnull().sum())

print("=" * 60)
print("Target Distribution")
print(df["target"].value_counts())

# ============================================================
# STEP 4 - Separate Features and Target
# ============================================================

X = df.drop("target", axis=1)

y = df["target"]

# ============================================================
# STEP 5 - Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# STEP 6 - Feature Scaling
# ============================================================

"""
PCA is sensitive to feature scales.

StandardScaler standardizes every feature so that:

Mean = 0
Standard Deviation = 1

Without scaling,
features with large values dominate PCA.
"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 7 - Logistic Regression WITHOUT PCA
# ============================================================

print("=" * 60)
print("Training Logistic Regression WITHOUT PCA")

model_without_pca = LogisticRegression(max_iter=5000)

model_without_pca.fit(
    X_train_scaled,
    y_train
)

y_pred_without = model_without_pca.predict(
    X_test_scaled
)

accuracy_without = accuracy_score(
    y_test,
    y_pred_without
)

print("Accuracy Without PCA :", accuracy_without)

# ============================================================
# STEP 8 - Apply PCA
# ============================================================

"""
n_components=None

Keep ALL principal components.

This allows us to calculate how much
variance every component explains.
"""

pca = PCA()

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

# ============================================================
# STEP 9 - Explained Variance
# ============================================================

print("=" * 60)
print("Explained Variance Ratio")

print(pca.explained_variance_ratio_)

# ============================================================
# STEP 10 - Cumulative Explained Variance
# ============================================================

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print("=" * 60)
print("Cumulative Explained Variance")

print(cumulative_variance)

# ============================================================
# STEP 11 - Plot Cumulative Variance
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance,
    marker='o'
)

plt.xlabel("Number of Principal Components")

plt.ylabel("Cumulative Explained Variance")

plt.title("Explained Variance by PCA")

plt.grid(True)

plt.show()

# ============================================================
# STEP 12 - Keep 95% Information
# ============================================================

"""
n_components=0.95

Automatically choose the minimum number
of principal components that preserve
95% of the information.
"""

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

print("=" * 60)
print("Original Features :", X_train.shape[1])

print("Selected Components :", pca.n_components_)

# ============================================================
# STEP 13 - Logistic Regression WITH PCA
# ============================================================

print("=" * 60)
print("Training Logistic Regression WITH PCA")

model_with_pca = LogisticRegression(max_iter=5000)

model_with_pca.fit(
    X_train_pca,
    y_train
)

y_pred_with = model_with_pca.predict(
    X_test_pca
)

accuracy_with = accuracy_score(
    y_test,
    y_pred_with
)

print("Accuracy With PCA :", accuracy_with)

# ============================================================
# STEP 14 - Compare Results
# ============================================================

print("=" * 60)

print(f"Accuracy Without PCA : {accuracy_without:.4f}")

print(f"Accuracy With PCA    : {accuracy_with:.4f}")

# ============================================================
# STEP 15 - Confusion Matrix
# ============================================================

print("=" * 60)
print("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred_with
)

print(cm)

# ============================================================
# STEP 16 - Classification Report
# ============================================================

print("=" * 60)
print("Classification Report")

print(
    classification_report(
        y_test,
        y_pred_with
    )
)

# ============================================================
# STEP 17 - 2D Visualization using PCA
# ============================================================

"""
Now reduce the dataset to only
2 principal components
for visualization.
"""

pca_visual = PCA(
    n_components=2
)

X_visual = pca_visual.fit_transform(
    X_train_scaled
)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_visual[:,0],
    X_visual[:,1],
    c=y_train,
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA Visualization")

plt.colorbar(scatter)

plt.show()

# ============================================================
# STEP 18 - Project Completed
# ============================================================

print("=" * 60)
print("Project Completed Successfully")
print("=" * 60)