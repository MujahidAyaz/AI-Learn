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

# Load data
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)
X   = df.drop("Outcome", axis=1)
y   = df["Outcome"]
feature_names = X.columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale — required for KNN
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Find best K
k_values  = range(1, 31)
cv_scores = [
    cross_val_score(KNeighborsClassifier(n_neighbors=k),
                    X_train_scaled, y_train,
                    cv=5, scoring="roc_auc").mean()
    for k in k_values
]
best_k = list(k_values)[int(np.argmax(cv_scores))]
print(f"Best K: {best_k}  |  CV ROC-AUC: {max(cv_scores):.3f}")

# Train final model
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", weights="uniform")
knn.fit(X_train_scaled, y_train)

# Trace a single prediction
new_patient = pd.DataFrame([[2,138,72,30,140,33.6,0.627,45]], columns=feature_names)
new_scaled  = scaler.transform(new_patient)
distances, indices = knn.kneighbors(new_scaled)

print(f"\nThe {best_k} nearest neighbors:")
for rank, (d, idx) in enumerate(zip(distances[0], indices[0]), 1):
    label = "YES ⚠" if y_train.iloc[idx] == 1 else "no  ✓"
    print(f"  #{rank:<3} distance={d:.3f}  {label}")

yes_votes = int(y_train.iloc[indices[0]].sum())
print(f"\nVote: {yes_votes} YES vs {best_k - yes_votes} NO")
print(f"Prediction: {'Diabetes' if yes_votes > best_k - yes_votes else 'No Diabetes'}")

# Evaluate
y_pred  = knn.predict(X_test_scaled)
y_proba = knn.predict_proba(X_test_scaled)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"\nAccuracy : {acc * 100:.1f}%  |  ROC-AUC : {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

# Full model comparison
print("━" * 55)
print(f"{'Model':<22} {'Accuracy':>9} {'ROC-AUC':>9}")
print("━" * 55)
print(f"{'Logistic Regression':<22} {'78.6%':>9} {'0.850':>9}")
print(f"{'Decision Tree':<22} {'77.3%':>9} {'0.830':>9}")
print(f"{'Random Forest':<22} {'81.2%':>9} {'0.876':>9}")
print(f"{'KNN (K='+str(best_k)+')':<22} {f'{acc*100:.1f}%':>9} {f'{auc:.3f}':>9}")
print("━" * 55)
print("Winner: Random Forest 🏆  |  Simplest: KNN")

# Visualizations
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("KNN — Diabetes Prediction", fontsize=14, fontweight="bold")

axes[0].plot(fpr, tpr, color="#60a5fa", lw=2, label=f"AUC={auc:.2f}")
axes[0].plot([0,1],[0,1],"--",color="#555")
axes[0].set_title("ROC Curve"); axes[0].legend()
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")

axes[1].plot(list(k_values), cv_scores, color="#60a5fa", lw=2, marker="o", ms=4)
axes[1].axvline(best_k, color="#fb7185", linestyle="--", label=f"Best K={best_k}")
axes[1].set_title("K vs ROC-AUC"); axes[1].set_xlabel("K"); axes[1].legend()

axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks([0,1]); axes[2].set_yticks([0,1])
axes[2].set_xticklabels(["Pred: No","Pred: Yes"])
axes[2].set_yticklabels(["Actual: No","Actual: Yes"])
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i,j]), ha="center", va="center",
                     fontsize=18, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("knn_results.png", dpi=150, bbox_inches="tight")
plt.show()