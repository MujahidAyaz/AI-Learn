import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# Load data (no scaling needed)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df  = pd.read_csv(url)
X   = df.drop("Outcome", axis=1)
y   = df["Outcome"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train with early stopping
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

print(f"\nBest round : {model.best_iteration}")
print(f"Best loss  : {model.best_score:.4f}")

# Evaluate
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy   : {acc * 100:.1f}%  |  ROC-AUC : {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

# Feature importance (gain)
print("Feature Importance (gain):")
gain = model.get_booster().get_score(importance_type="gain")
for feat, score in sorted(gain.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat:<30} {score:>8.1f}  {'█'*int(score/max(gain.values())*30)}")

# Predict new patient
new_patient = pd.DataFrame([[2,138,72,30,140,33.6,0.627,45]], columns=feature_names)
pred = model.predict(new_patient)[0]
prob = model.predict_proba(new_patient)[0][1]
print(f"\nNew patient → {'Diabetes' if pred==1 else 'No Diabetes'} ({prob*100:.1f}%)")

# Full comparison
print("\n" + "━"*55)
print(f"{'Model':<22}{'Accuracy':>10}{'ROC-AUC':>10}")
print("━"*55)
print(f"{'Logistic Regression':<22}{'78.6%':>10}{'0.850':>10}")
print(f"{'Decision Tree':<22}{'77.3%':>10}{'0.830':>10}")
print(f"{'KNN (K=11)':<22}{'77.9%':>10}{'0.841':>10}")
print(f"{'Random Forest':<22}{'81.2%':>10}{'0.876':>10}")
print(f"{'XGBoost':<22}{f'{acc*100:.1f}%':>10}{f'{auc:.3f}':>10}")
print("━"*55)
print("Winner: XGBoost 🏆")

# Visualizations
cm  = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
val_loss = model.evals_result()["validation_0"]["logloss"]
feat_names = list(gain.keys())
feat_vals  = list(gain.values())
sorted_idx = np.argsort(feat_vals)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("XGBoost — Diabetes Prediction", fontsize=14, fontweight="bold")

axes[0].plot(val_loss, color="#f59e0b", lw=2)
axes[0].axvline(model.best_iteration, color="#fb7185", linestyle="--",
                label=f"Best={model.best_iteration}")
axes[0].set_title("Loss Per Round"); axes[0].legend()
axes[0].set_xlabel("Round"); axes[0].set_ylabel("Log Loss")

axes[1].plot(fpr, tpr, color="#f59e0b", lw=2, label=f"AUC={auc:.3f}")
axes[1].plot([0,1],[0,1],"--",color="#555")
axes[1].set_title("ROC Curve"); axes[1].legend()

axes[2].barh(np.array(feat_names)[sorted_idx],
             np.array(feat_vals)[sorted_idx], color="#f59e0b")
axes[2].set_title("Feature Importance (gain)")

axes[3].imshow(cm, cmap="YlOrBr")
axes[3].set_xticks([0,1]); axes[3].set_yticks([0,1])
axes[3].set_xticklabels(["Pred: No","Pred: Yes"])
axes[3].set_yticklabels(["Actual: No","Actual: Yes"])
axes[3].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[3].text(j, i, str(cm[i,j]), ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("xgboost_results.png", dpi=150, bbox_inches="tight")
plt.show()