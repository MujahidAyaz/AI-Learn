# ============================================================
#  STAGE 7-10 — SPLIT, TRAIN, TUNE, EVALUATE
#  Email/SMS Spam Classifier — SVM
# ============================================================
#
# PLAIN ENGLISH:
# We pick SVM for this project because it tends to perform very
# well on high-dimensional sparse data — and text turned into
# TF-IDF vectors is exactly that: thousands of mostly-zero
# columns (one per word). SVM finds the widest possible margin
# between "spam" and "ham" in that word-space.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

df = pd.read_csv("data/cleaned_spam_data.csv")

X_text = df["clean_message"]
y = df["label_num"]


# ────────────────────────────────────────────────────────────
# STAGE 7 — TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# Same idea as every previous project — 80% to learn, 20% held
# back for an honest final exam. We use stratify=y this time,
# which guarantees both the train and test sets keep the same
# 87/13 ham/spam ratio as the full dataset. Without stratify,
# a random split could accidentally put very few spam messages
# in the test set, making evaluation unreliable.

print("=" * 55)
print("STAGE 7 — TRAIN / TEST SPLIT")
print("=" * 55)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training messages : {len(X_train_text)}")
print(f"Testing  messages : {len(X_test_text)}")
print(f"Train spam ratio  : {y_train.mean():.1%}")
print(f"Test  spam ratio  : {y_test.mean():.1%}")
print("(stratify keeps the ratio identical in both sets)")


# ────────────────────────────────────────────────────────────
# TF-IDF — turning text into numbers
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# Models can't read words — they need numbers. TF-IDF
# (Term Frequency - Inverse Document Frequency) converts each
# message into a row of numbers, one column per word.
#
#   Term Frequency      -> how often a word appears in THIS message
#   Inverse Doc Frequency -> how RARE that word is across ALL messages
#
# Common words like "the" get a low score (everyone uses them,
# not informative). Rare, spam-flavored words like "winner" or
# "claim" get a high score when they DO appear — they're more
# informative about what the message actually is.
#
# ngram_range=(1,2) -> capture single words AND two-word phrases
#   (e.g. "free" and "free entry" are both captured separately)
# max_features=3000  -> keep only the 3000 most useful word columns
# min_df=2            -> ignore words that appear in only 1 message
#                        (too rare to be a reliable pattern)

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train_text)   # learn vocabulary + transform
X_test_tfidf  = tfidf.transform(X_test_text)        # transform only (no relearning!)

print()
print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"  -> {X_train_tfidf.shape[0]} messages x {X_train_tfidf.shape[1]} word features")


# ────────────────────────────────────────────────────────────
# STAGE 8 — MODEL TRAINING + HYPERPARAMETER TUNING
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# We use LinearSVC — the linear-kernel version of SVM, which
# is fast and works extremely well on text data like this.
#
# class_weight="balanced" -> automatically gives the minority
#   class (spam, 13%) more importance during training, so the
#   model doesn't just lazily predict "ham" every time.
#
# We don't guess the C parameter — we test several values with
# GridSearchCV and 5-fold cross-validation, picking whichever
# C scores best on F1 (a balance of precision and recall, more
# meaningful than accuracy on imbalanced data like this).

print()
print("=" * 55)
print("STAGE 8 — MODEL TRAINING + HYPERPARAMETER TUNING")
print("=" * 55)

param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=5000, random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_tfidf, y_train)

print("Tested C values  :", param_grid["C"])
print(f"Best C            : {grid_search.best_params_['C']}")
print(f"Best CV F1 score  : {grid_search.best_score_:.4f}")

print()
print("Full grid results:")
results_df = pd.DataFrame(grid_search.cv_results_)[
    ["param_C", "mean_test_score", "std_test_score"]
]
print(results_df.to_string(index=False))

best_svm = grid_search.best_estimator_

# LinearSVC doesn't output probabilities by default — only a
# hard yes/no decision. CalibratedClassifierCV wraps it to also
# produce probability estimates, which we want for confidence
# scores in the final prediction pipeline.
print()
print("Calibrating model to produce probability estimates...")
final_model = CalibratedClassifierCV(best_svm, cv=5)
final_model.fit(X_train_tfidf, y_train)
print("Calibration complete.")


# ────────────────────────────────────────────────────────────
# STAGE 9 — EVALUATION
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# Final exam on the 20% the model has never seen.
# On imbalanced data, accuracy alone lies — a model that always
# guesses "ham" would already score 87%. We check precision,
# recall, and F1 specifically for the spam class.
#
#   Precision (spam) -> of messages flagged spam, how many really are?
#                        Low precision = annoying false alarms,
#                        real emails get blocked.
#   Recall (spam)    -> of all real spam, how many did we catch?
#                        Low recall = spam slips through to the inbox.

print()
print("=" * 55)
print("STAGE 9 — EVALUATION ON HELD-OUT TEST SET")
print("=" * 55)

y_pred  = final_model.predict(X_test_tfidf)
y_proba = final_model.predict_proba(X_test_tfidf)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

print(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision : {prec:.4f}  (of predicted spam, how many were really spam)")
print(f"Recall    : {rec:.4f}  (of all real spam, how many we caught)")
print(f"F1 Score  : {f1:.4f}  (balance of precision and recall)")
print(f"ROC-AUC   : {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  True Ham  correctly kept    : {cm[0][0]}")
print(f"  Ham wrongly flagged spam    : {cm[0][1]}  <- false alarms, annoying")
print(f"  Spam that slipped through   : {cm[1][0]}  <- missed spam, risky")
print(f"  Spam correctly caught       : {cm[1][1]}")

# Visualize
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("SVM Spam Classifier — Evaluation", fontsize=14, fontweight="bold")

axes[0].plot(fpr, tpr, color="#a78bfa", lw=2, label=f"AUC={auc:.3f}")
axes[0].plot([0,1],[0,1],"--",color="#555")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(results_df["param_C"], results_df["mean_test_score"],
             color="#a78bfa", marker="o", lw=2)
axes[1].set_xscale("log")
axes[1].set_title("Hyperparameter Search\n(C vs F1 score)")
axes[1].set_xlabel("C (log scale)")
axes[1].set_ylabel("Cross-Val F1")

axes[2].imshow(cm, cmap="Purples")
axes[2].set_xticks([0,1]); axes[2].set_yticks([0,1])
axes[2].set_xticklabels(["Pred: Ham","Pred: Spam"])
axes[2].set_yticklabels(["Actual: Ham","Actual: Spam"])
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i,j]), ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Evaluation charts saved to: evaluation_results.png")


# ────────────────────────────────────────────────────────────
# STAGE 10 — MODEL SAVING
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# Training takes time. We don't want to retrain every time we
# want to make a prediction. joblib saves the trained model
# and the TF-IDF vectorizer to disk as .pkl files — we can load
# them instantly later without retraining anything.
#
# IMPORTANT: we must save the vectorizer too, not just the
# model. The vectorizer is what turns new raw text into the
# same numeric format the model was trained on. Using a
# mismatched or freshly-refit vectorizer would silently break
# every future prediction.

print()
print("=" * 55)
print("STAGE 10 — MODEL SAVING")
print("=" * 55)

joblib.dump(final_model, "models/svm_spam_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("Saved: models/svm_spam_model.pkl")
print("Saved: models/tfidf_vectorizer.pkl")
print("Both files are required together for predictions.")
