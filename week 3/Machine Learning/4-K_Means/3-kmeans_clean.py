import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans
from sklearn.metrics       import silhouette_score

# Load data
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# Scale (required for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find best K
inertias, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"Best K: {best_k}  (silhouette={max(sil_scores):.4f})")

# Train final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Interpret clusters
summary = df.groupby("Cluster")[["Annual Income (k$)","Spending Score (1-100)"]].mean().round(1)
summary["Count"] = df["Cluster"].value_counts().sort_index()
print(summary)

cluster_names = {
    0: "Average Joe", 1: "VIP (high income, high spend)",
    2: "Impulsive (low income, high spend)",
    3: "Careful (high income, low spend)",
    4: "Budget (low income, low spend)",
}

# Predict new customer
new_customer = np.array([[75, 85]])
new_scaled = scaler.transform(new_customer)
cluster = kmeans.predict(new_scaled)[0]
print(f"\nNew customer (income=75k, score=85) -> Cluster {cluster}: {cluster_names.get(cluster,'')}")

# Visualize
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
colors = ["#818cf8", "#fb7185", "#4ade80", "#fbbf24", "#c084fc"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("K-Means — Customer Segmentation", fontsize=14, fontweight="bold")

axes[0].plot(list(K_range), inertias, color="#818cf8", marker="o", lw=2)
axes[0].axvline(best_k, color="#fb7185", linestyle="--", label=f"K={best_k}")
axes[0].set_title("Elbow Method"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(list(K_range), sil_scores, color="#2dd4bf", marker="o", lw=2)
axes[1].axvline(best_k, color="#fb7185", linestyle="--", label=f"K={best_k}")
axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("K")
axes[1].legend()

for c in range(best_k):
    mask = df["Cluster"] == c
    axes[2].scatter(df[mask]["Annual Income (k$)"], df[mask]["Spending Score (1-100)"],
                     c=colors[c % len(colors)], label=f"Cluster {c}", alpha=0.7, s=40)
axes[2].scatter(centroids_orig[:,0], centroids_orig[:,1], c="black", marker="X", s=200, label="Centroids")
axes[2].set_title("Final Clusters"); axes[2].set_xlabel("Annual Income (k$)")
axes[2].set_ylabel("Spending Score (1-100)"); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("kmeans_results.png", dpi=150, bbox_inches="tight")
plt.show()
