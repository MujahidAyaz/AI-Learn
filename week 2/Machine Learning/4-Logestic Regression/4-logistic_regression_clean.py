import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# Load data
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)

# Split features and label
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train model
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.1f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
print()
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# Predict new patient
feature_names = X.columns.tolist()
new_patient   = pd.DataFrame([[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
                               columns=feature_names)
prob = model.predict_proba(scaler.transform(new_patient))[0][1]
pred = model.predict(scaler.transform(new_patient))[0]
print(f"New patient → {'Diabetes' if pred == 1 else 'No Diabetes'} ({prob*100:.1f}% probability)")

# Visualizations
weights = model.coef_[0]
cm      = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc     = roc_auc_score(y_test, y_proba)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Logistic Regression — Diabetes Prediction", fontsize=14, fontweight="bold")

axes[0].plot(fpr, tpr, color="#3fb950", lw=2, label=f"AUC = {auc:.2f}")
axes[0].plot([0, 1], [0, 1], "--", color="#555")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

sorted_idx = np.argsort(weights)
colors = ["#ff7b72" if w < 0 else "#3fb950" for w in weights[sorted_idx]]
axes[1].barh(np.array(feature_names)[sorted_idx], weights[sorted_idx], color=colors)
axes[1].axvline(0, color="#aaa", lw=0.8)
axes[1].set_title("Feature Weights")
axes[1].set_xlabel("Weight")

axes[2].imshow(cm, cmap="Greens")
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
plt.savefig("logistic_regression_results.png", dpi=150, bbox_inches="tight")
plt.show()
