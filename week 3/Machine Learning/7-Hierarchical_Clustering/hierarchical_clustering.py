"""
===========================================================
PROJECT: Hierarchical Clustering on Mall Customers Dataset
Author : Mujahid Ayaz
Algorithm : Agglomerative Hierarchical Clustering
===========================================================

Objective:
Group mall customers into different segments based on
their Annual Income and Spending Score.

Dataset Source:
https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv
"""

# ==========================================================
# Import Libraries
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# ==========================================================
# Load Dataset
# ==========================================================

url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"

df = pd.read_csv(url)

print("=" * 60)
print("First 5 Rows")
print("=" * 60)

print(df.head())

# ==========================================================
# Dataset Information
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# ==========================================================
# Select Features
# ==========================================================

# Annual Income
# Spending Score

X = df.iloc[:, [3, 4]].values

# ==========================================================
# Create Dendrogram
# ==========================================================

plt.figure(figsize=(12,7))

dendrogram(
    linkage(X,
            method="ward")
)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")

plt.show()

# ==========================================================
# Build Hierarchical Clustering Model
# ==========================================================

# From dendrogram we choose 5 clusters

model = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

y_pred = model.fit_predict(X)

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(10,7))

colors = [
    "red",
    "blue",
    "green",
    "cyan",
    "magenta"
]

for i in range(5):
    plt.scatter(
        X[y_pred == i, 0],
        X[y_pred == i, 1],
        s=70,
        color=colors[i],
        label=f"Cluster {i+1}"
    )

plt.title("Customer Segments (Hierarchical Clustering)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()

# ==========================================================
# Cluster Summary
# ==========================================================

df["Cluster"] = y_pred

print("\nCluster Counts")
print(df["Cluster"].value_counts())

print("\nAverage Values Per Cluster")
print(
    df.groupby("Cluster")[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ].mean()
)

print("\nProject Completed Successfully!")