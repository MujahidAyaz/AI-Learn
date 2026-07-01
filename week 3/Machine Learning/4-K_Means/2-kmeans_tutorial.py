# ============================================================
#  K-MEANS CLUSTERING — Complete Beginner Tutorial
#  Problem: Segment mall customers into natural groups
#  Dataset: Mall Customer Segmentation Data (200 customers)
# ============================================================
#
# BIG DIFFERENCE FROM EVERYTHING BEFORE:
# This is UNSUPERVISED learning. There is no "Outcome" column,
# no right answer to learn from. We're not predicting anything —
# we're discovering natural groups hiding in the data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans
from sklearn.metrics       import silhouette_score


# ────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ────────────────────────────────────────────────────────────
# Real dataset: 200 mall customers with income and spending
# behavior. No labels — nobody has told us what "type" of
# customer each person is. That's exactly what we want to find.

df = pd.read_csv("Mall_Customers.csv")

print("=" * 55)
print("STEP 1 — DATA LOADED")
print("=" * 55)
print(f"Total customers : {len(df)}")
print(df.head())
print()
print(df.describe().round(1))


# ────────────────────────────────────────────────────────────
# STEP 2 — SELECT FEATURES & SCALE
# ────────────────────────────────────────────────────────────
# We'll cluster customers based on two features:
#   - Annual Income (k$)
#   - Spending Score (1-100, store-assigned based on behavior)
#
# K-Means measures DISTANCE between points — just like KNN.
# Income ranges 15-137, Spending Score ranges 1-99. Without
# scaling, income would dominate every distance calculation
# just because its numbers are bigger. We MUST scale.

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print()
print("=" * 55)
print("STEP 2 — FEATURES SCALED")
print("=" * 55)
print("Before scaling (first row):", X[0])
print("After  scaling (first row):", X_scaled[0].round(3))


# ────────────────────────────────────────────────────────────
# STEP 3 — FIND THE BEST K (Elbow Method + Silhouette Score)
# ────────────────────────────────────────────────────────────
# How many natural groups exist in this data? We don't know
# ahead of time — that's the whole point of unsupervised
# learning. We test K=2 through K=10 and look at two signals:
#
#   Inertia -> total distance of every point to its own
#              centroid. Always drops as K increases (more
#              clusters = tighter fit) but with diminishing
#              returns. We want the "elbow" where it stops
#              dropping fast.
#
#   Silhouette Score -> how well-separated the clusters are
#              (ranges -1 to 1, higher is better). More
#              reliable than inertia alone for picking K.

print()
print("=" * 55)
print("STEP 3 — FINDING BEST K")
print("=" * 55)

inertias   = []
sil_scores = []
K_range    = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}  inertia={km.inertia_:.2f}  silhouette={sil_scores[-1]:.4f}")

best_k = list(K_range)[int(np.argmax(sil_scores))]
print()
print(f"Best K by silhouette score: {best_k}")

# Plot elbow curve and silhouette curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, color="#818cf8", marker="o", lw=2)
axes[0].axvline(best_k, color="#fb7185", linestyle="--", label=f"Chosen K={best_k}")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K (number of clusters)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(list(K_range), sil_scores, color="#2dd4bf", marker="o", lw=2)
axes[1].axvline(best_k, color="#fb7185", linestyle="--", label=f"Best K={best_k}")
axes[1].set_title("Silhouette Score by K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
print("Elbow chart saved to: kmeans_elbow.png")


# ────────────────────────────────────────────────────────────
# STEP 4 — TRAIN THE FINAL MODEL
# ────────────────────────────────────────────────────────────
# n_clusters=5    -> the K we found above
# n_init=10       -> run the whole algorithm 10 times with
#                    different random starting centroids, keep
#                    the best result (avoids bad random luck)
# random_state=42 -> reproducible results

print()
print("=" * 55)
print(f"STEP 4 — TRAINING FINAL MODEL (K={best_k})")
print("=" * 55)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels

print("Clustering complete!")
print(f"Final inertia: {kmeans.inertia_:.2f}")
print()
print("Customers per cluster:")
print(df["Cluster"].value_counts().sort_index())


# ────────────────────────────────────────────────────────────
# STEP 5 — INTERPRET THE CLUSTERS
# ────────────────────────────────────────────────────────────
# K-Means gives us cluster NUMBERS (0, 1, 2, 3, 4) — it doesn't
# know or care what they MEAN. That's our job. We look at the
# average income and spending score per cluster and give each
# one a human-readable name based on the pattern we see.

print()
print("=" * 55)
print("STEP 5 — INTERPRETING THE CLUSTERS")
print("=" * 55)

cluster_summary = df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean().round(1)
cluster_summary["Count"] = df["Cluster"].value_counts().sort_index()

print(cluster_summary)

# Business-friendly names based on the actual patterns found
cluster_names = {
    0: "Average Joe — moderate income, moderate spending",
    1: "VIP — high income, high spending (target for premium offers)",
    2: "Impulsive — low income, high spending (risk of overspending)",
    3: "Careful — high income, low spending (untapped potential)",
    4: "Budget — low income, low spending (price-sensitive)",
}

print()
print("Business interpretation:")
for cluster_id, name in cluster_names.items():
    count = (df["Cluster"] == cluster_id).sum()
    print(f"  Cluster {cluster_id} ({count} customers): {name}")


# ────────────────────────────────────────────────────────────
# STEP 6 — VISUALIZE THE FINAL CLUSTERS
# ────────────────────────────────────────────────────────────

print()
print("=" * 55)
print("STEP 6 — VISUALIZING CLUSTERS")
print("=" * 55)

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
colors = ["#818cf8", "#fb7185", "#4ade80", "#fbbf24", "#c084fc"]

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(best_k):
    mask = df["Cluster"] == c
    ax.scatter(df[mask]["Annual Income (k$)"], df[mask]["Spending Score (1-100)"],
               c=colors[c % len(colors)], label=f"Cluster {c}", alpha=0.7, s=50)

ax.scatter(centroids_original[:, 0], centroids_original[:, 1],
           c="black", marker="X", s=250, label="Centroids", edgecolors="white", linewidth=1.5)

ax.set_title(f"Customer Segments (K={best_k})", fontsize=13, fontweight="bold")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()

plt.tight_layout()
plt.savefig("kmeans_final_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print("Final cluster chart saved to: kmeans_final_clusters.png")


# ────────────────────────────────────────────────────────────
# STEP 7 — ASSIGN A NEW CUSTOMER TO A CLUSTER
# ────────────────────────────────────────────────────────────
# A new customer walks in. We don't know their "type" — but we
# can predict which existing cluster they're closest to.

print()
print("=" * 55)
print("STEP 7 — CLASSIFY A NEW CUSTOMER")
print("=" * 55)

new_customer = np.array([[75, 85]])  # income=75k, spending score=85
new_customer_scaled = scaler.transform(new_customer)

predicted_cluster = kmeans.predict(new_customer_scaled)[0]

print(f"New customer: Income=$75k, Spending Score=85")
print(f"Assigned cluster : {predicted_cluster}")
print(f"Segment          : {cluster_names[predicted_cluster]}")

print()
print("Note: K-Means has no 'confidence score' like classifiers do —")
print("it simply assigns the point to its nearest centroid.")
