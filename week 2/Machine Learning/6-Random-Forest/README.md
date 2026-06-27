# Random Forest — Diabetes Prediction

A beginner-friendly implementation of a Random Forest to predict whether a patient has diabetes, using the Pima Indians Diabetes Dataset.

## What this project covers

- Loading a real-world dataset
- Train/Test split (no scaling needed!)
- Training 100 decision trees with bagging and random feature selection
- OOB Score — a free accuracy estimate with no test set needed
- Seeing how majority voting works across all 100 trees
- Evaluating with Accuracy, Precision, Recall, F1, ROC-AUC
- Feature importance averaged across all 100 trees
- Cross-validation — a more reliable accuracy test than one split
- Comparing Random Forest vs Logistic Regression vs Decision Tree

## Files

| File | Description |
|---|---|
| `random_forest_tutorial.py` | Step-by-step code with detailed plain-English comments |
| `random_forest_clean.py` | The complete code in one clean script (no comments) |

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python random_forest_tutorial.py
```

## Dataset

[Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv) — 768 patients, 8 features, binary outcome (diabetes yes/no).

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 78.6% | 0.850 |
| Decision Tree | 77.3% | 0.830 |
| **Random Forest** | **81.2%** | **0.876** |

Random Forest wins — ensemble learning beats any single model.

## Key Concepts

| Concept | Plain English |
|---|---|
| **n_estimators** | How many trees to build (100 here) |
| **Bagging** | Each tree trains on a random sample of patients |
| **max_features** | Each tree only sees √8 ≈ 3 random features per split |
| **Majority Voting** | 67 trees say YES, 33 say NO → Final: YES |
| **OOB Score** | Free accuracy test using each tree's leftover data |
| **Cross-Validation** | 5 different train/test splits averaged — more reliable than one |
| **Feature Importance** | Which features mattered most, averaged over all 100 trees |
