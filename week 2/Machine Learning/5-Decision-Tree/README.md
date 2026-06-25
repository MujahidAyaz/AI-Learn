# Decision Tree — Diabetes Prediction

A beginner-friendly implementation of a Decision Tree to predict whether a patient has diabetes, using the Pima Indians Diabetes Dataset.

## What this project covers

- Loading a real-world dataset
- Train/Test split (no scaling needed!)
- Training an unpruned tree to see overfitting in action
- Pruning with max_depth to fix overfitting
- Reading the tree's exact decision rules
- Visualizing the tree as a diagram
- Feature importance — which clues matter most
- Predicting on a new patient

## Files

| File | Description |
|---|---|
| `decision_tree_tutorial.py` | Step-by-step code with detailed plain-English comments |
| `decision_tree_clean.py` | The complete code in one clean script (no comments) |

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python decision_tree_tutorial.py
```

## Dataset

[Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv) — 768 patients, 8 features, binary outcome (diabetes yes/no).

## Results

| Model | Train Accuracy | Test Accuracy | ROC-AUC |
|---|---|---|---|
| Unpruned Tree (overfit) | 100% | 69.5% | — |
| Pruned Tree (max_depth=4) | 80.3% | 77.3% | 0.83 |

## Key Concepts

| Concept | Plain English |
|---|---|
| **Node** | A yes/no question the tree asks |
| **Leaf** | The final answer — no more questions |
| **Gini Impurity** | How mixed a group is (lower = better split) |
| **Overfitting** | Tree memorized training data, fails on new data |
| **max_depth** | Limits how deep the tree grows to prevent overfitting |
| **Feature Importance** | How much each feature was used in decisions |
