# Email/SMS Spam Classifier — SVM

An end-to-end machine learning project that classifies text messages as **spam** or **ham** (not spam), built following the complete ML development lifecycle — from problem definition to a production-ready prediction pipeline.

This is not a toy example. It runs on a real, publicly available dataset of 5,572 actual text messages, and every number in this README comes from an actual run of the code in this repo.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.07% |
| Precision (spam) | 94.40% |
| Recall (spam) | 90.08% |
| F1-score (spam) | 92.19% |
| ROC-AUC | 0.9927 |

Confusion Matrix (1,034 test messages, never seen during training):

| | Predicted: Ham | Predicted: Spam |
|---|---|---|
| **Actual: Ham** | 896 | 7 |
| **Actual: Spam** | 13 | 118 |

Out of every 131 real spam messages in the test set, **118 were caught** and only **13 slipped through**. Out of 903 real messages, only **7 were wrongly flagged**.

---

## The Development Lifecycle

This project follows the complete ML lifecycle rather than jumping straight to model code:

```
Problem Understanding
        ↓
Dataset Collection
        ↓
Exploratory Data Analysis
        ↓
Cleaning
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Evaluation
        ↓
Model Saving
        ↓
Prediction Pipeline
```

Each stage lives in its own numbered file, runnable independently and in order.

| File | Lifecycle Stage(s) | What it does |
|---|---|---|
| `01_02_problem_and_dataset.py` | Problem Understanding, Dataset Collection | Defines the ML problem and loads the raw dataset |
| `03_eda.py` | Exploratory Data Analysis | Visualizes class balance, message length, spam vocabulary |
| `04_cleaning_preprocessing_features.py` | Cleaning, Preprocessing, Feature Engineering | Deduplicates, normalizes text, engineers numeric signals |
| `05_train_tune_evaluate.py` | Split, Training, Tuning, Evaluation, Saving | TF-IDF + SVM, GridSearchCV, full metrics, saves model |
| `06_prediction_pipeline.py` | Prediction Pipeline | Loads saved model, classifies new messages |

---

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn joblib

# Run stages in order (or skip to 06 using the pre-trained model included)
python 01_02_problem_and_dataset.py
python 03_eda.py
python 04_cleaning_preprocessing_features.py
python 05_train_tune_evaluate.py
python 06_prediction_pipeline.py
```

The trained model (`models/svm_spam_model.pkl`) and vectorizer (`models/tfidf_vectorizer.pkl`) are included, so you can run `06_prediction_pipeline.py` directly without retraining.

---

## Dataset

**SMS Spam Collection** — 5,572 real text messages, manually labeled spam/ham.
Source: UCI Machine Learning Repository / public SMS spam dataset.

- Format: tab-separated, two columns (`label`, `message`)
- Class balance: ~87% ham, ~13% spam (realistic, imbalanced — not artificially balanced)
- After deduplication: 5,169 unique messages

---

## Why SVM?

SVM (Support Vector Machine) was chosen specifically for this problem because:

- Text converted to TF-IDF produces high-dimensional, sparse data (thousands of word-columns, mostly zero) — exactly the kind of data SVM handles very well
- `LinearSVC` is fast even with thousands of features
- The maximum-margin decision boundary tends to generalize well on this type of data, especially with proper `class_weight="balanced"` handling for the minority spam class

---

## Pipeline Details

### Text Cleaning
- Lowercased, URLs replaced with `URL` token, long digit sequences replaced with `PHONENUM` token, punctuation stripped, whitespace collapsed
- Edge case handled: messages that are only emoji/symbols (e.g. `:)`) become empty after cleaning — these are mapped to a safe placeholder instead of silently becoming invalid data

### Feature Extraction — TF-IDF
- `max_features=3000`, `ngram_range=(1,2)` (captures both single words and two-word phrases), `min_df=2`, English stop words removed

### Hyperparameter Tuning
- `GridSearchCV` over `C = [0.01, 0.1, 1, 10, 100]`, 5-fold cross-validation, scored on F1
- Best: **C=1**, CV F1 = 0.919

### Handling Class Imbalance
- `class_weight="balanced"` in the SVM automatically up-weights the minority (spam) class during training
- `stratify=y` in the train/test split keeps the same spam ratio in both sets
- F1-score used for model selection instead of raw accuracy, which would be misleading on imbalanced data

### Model Calibration
- `LinearSVC` outputs hard predictions only, no probabilities. Wrapped in `CalibratedClassifierCV` to produce confidence scores usable in the prediction pipeline.

---

## Example Predictions

```
⚠ SPAM   77.4%   Congratulations! You've won a $1000 Walmart gift card...
✓ HAM     0.4%   Hey, are we still meeting for lunch tomorrow at 1pm?
⚠ SPAM   93.7%   URGENT: Your account has been suspended. Verify...
✓ HAM     0.1%   Mom said dinner is ready, come downstairs
⚠ SPAM   96.5%   FREE entry in our weekly competition! Text WIN to 80082!!!
✓ HAM     3.5%   Can you send me the report before end of day?
```

---

## Project Structure

```
email-spam-classifier/
├── README.md
├── 01_02_problem_and_dataset.py
├── 03_eda.py
├── 04_cleaning_preprocessing_features.py
├── 05_train_tune_evaluate.py
├── 06_prediction_pipeline.py
├── eda_overview.png
├── evaluation_results.png
├── data/
│   ├── sms_spam_collection.tsv
│   └── cleaned_spam_data.csv
└── models/
    ├── svm_spam_model.pkl
    └── tfidf_vectorizer.pkl
```

## Future Improvements
- Try other kernels (RBF) and compare against the linear baseline
- Add a simple Flask/FastAPI wrapper around `06_prediction_pipeline.py` to serve predictions over HTTP
- Expand feature engineering (engineered numeric features like message length and capital ratio are currently computed but not yet fed into the model alongside TF-IDF)
