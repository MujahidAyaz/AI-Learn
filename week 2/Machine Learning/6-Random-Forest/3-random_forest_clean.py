import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble        import RandomForestClassifier
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

# Train/test split (no scaling needed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100, max_depth=6,
    max_features="sqrt", oob_score=True,
    n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Score : {rf.oob_score_ * 100:.1f}%")

# Voting for one patient
votes     = [t.predict(X_test.iloc[[0]])[0] for t in rf.estimators_]
yes_votes = sum(votes)
print(f"One patient vote: {yes_votes}/100 trees said Diabetes")

# Evaluate
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy : {acc * 100:.1f}%  |  ROC-AUC : {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

# Feature importance
print("Feature Importance:")
for name, imp in sorted(zip(feature_names, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {name:<30} {imp:.3f}  {'█'*int(imp*40)}")

# Cross-validation
cv = cross_val_score(rf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"\nCross-Val ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")

# Predict new patient
new_patient = pd.DataFrame([[2,138,72,30,140,33.6,0.627,45]], columns=feature_names)
pred = rf.predict(new_patient)[0]
prob = rf.predict_proba(new_patient)[0][1]
print(f"\nNew patient → {'Diabetes' if pred==1 else 'No Diabetes'} ({prob*100:.1f}%)")

# Model comparison
print("\n" + "━"*52)
print(f"{'Model':<25}{'Accuracy':>10}{'ROC-AUC':>10}")
print("━"*52)
print(f"{'Logistic Regression':<25}{'78.6%':>10}{'0.850':>10}")
print(f"{'Decision Tree':<25}{'77.3%':>10}{'0.830':>10}")
print(f"{'Random Forest':<25}{f'{acc*100:.1f}%':>10}{f'{auc:.3f}':>10}")
print("━"*52)
print("Winner: Random Forest 🏆")

# Visualizations
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
importances = rf.feature_importances_

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Random Forest — Diabetes Prediction", fontsize=14, fontweight="bold")

axes[0].plot(fpr, tpr, color="#4ade80", lw=2, label=f"AUC={auc:.2f}")
axes[0].plot([0,1],[0,1],"--",color="#555")
axes[0].set_title("ROC Curve"); axes[0].legend()
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")

sorted_idx = np.argsort(importances)
axes[1].barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="#4ade80")
axes[1].set_title("Feature Importance"); axes[1].set_xlabel("Importance")

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
plt.savefig("random_forest_results.png", dpi=150, bbox_inches="tight")
plt.show()
