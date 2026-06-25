import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree         import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics      import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# Load data
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)
X   = df.drop("Outcome", axis=1)
y   = df["Outcome"]
feature_names = X.columns.tolist()

# Train/test split (no scaling needed for trees)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Overfit tree — to show the problem
tree_overfit = DecisionTreeClassifier(random_state=42)
tree_overfit.fit(X_train, y_train)
print(f"Overfit  → Train: {accuracy_score(y_train, tree_overfit.predict(X_train))*100:.1f}%  "
      f"Test: {accuracy_score(y_test, tree_overfit.predict(X_test))*100:.1f}%  "
      f"Depth: {tree_overfit.get_depth()}")

# Pruned tree — fix overfitting
tree = DecisionTreeClassifier(
    max_depth=4, min_samples_split=10,
    min_samples_leaf=5, criterion="gini", random_state=42
)
tree.fit(X_train, y_train)
print(f"Pruned   → Train: {accuracy_score(y_train, tree.predict(X_train))*100:.1f}%  "
      f"Test: {accuracy_score(y_test, tree.predict(X_test))*100:.1f}%  "
      f"Depth: {tree.get_depth()}")

# Read the rules
print("\nDecision Rules:")
print(export_text(tree, feature_names=feature_names))

# Evaluate
y_pred  = tree.predict(X_test)
y_proba = tree.predict_proba(X_test)[:, 1]
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

# Feature importance
print("Feature Importance:")
for name, imp in sorted(zip(feature_names, tree.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"  {name:<30} {imp:.3f}  {'█' * int(imp*40)}")

# Predict new patient
new_patient = pd.DataFrame([[2, 138, 72, 30, 140, 33.6, 0.627, 45]],
                            columns=feature_names)
pred = tree.predict(new_patient)[0]
prob = tree.predict_proba(new_patient)[0][1]
print(f"\nNew patient → {'Diabetes' if pred==1 else 'No Diabetes'} ({prob*100:.1f}%)")

# Visualize tree diagram
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(tree, feature_names=feature_names,
          class_names=["No Diabetes","Diabetes"],
          filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title("Decision Tree — Diabetes Prediction (max_depth=4)")
plt.tight_layout()
plt.savefig("decision_tree_diagram.png", dpi=150, bbox_inches="tight")
plt.show()

# Summary charts
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
importances = tree.feature_importances_

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Decision Tree — Diabetes Prediction", fontsize=14, fontweight="bold")

axes[0].plot(fpr, tpr, color="#52d48a", lw=2, label=f"AUC = {auc:.2f}")
axes[0].plot([0,1],[0,1],"--",color="#555")
axes[0].set_title("ROC Curve"); axes[0].legend()
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")

sorted_idx = np.argsort(importances)
axes[1].barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="#f5a623")
axes[1].set_title("Feature Importance"); axes[1].set_xlabel("Importance")

axes[2].imshow(cm, cmap="YlOrBr")
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
plt.savefig("decision_tree_results.png", dpi=150, bbox_inches="tight")
plt.show()
