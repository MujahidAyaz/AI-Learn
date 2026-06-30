# ============================================================
#  STAGE 1-2 — PROBLEM UNDERSTANDING & DATASET COLLECTION
#  Email/SMS Spam Classifier
# ============================================================
#
# This file is documentation-as-code — it doesn't run a pipeline,
# it records the decisions made before any code was written.
# In a real project this thinking would live in a README or a
# design doc, but it's included here as a script so the full
# lifecycle is traceable in order.


# ────────────────────────────────────────────────────────────
# STAGE 1 — PROBLEM UNDERSTANDING
# ────────────────────────────────────────────────────────────
#
# PLAIN ENGLISH:
# Before writing a single line of ML code, we need to be
# precise about what we're actually solving.
#
# Business problem:
#   Inboxes get flooded with spam — scams, phishing, ads.
#   We want a system that automatically flags incoming
#   messages as "spam" or "ham" (not spam) before a human
#   ever sees them.
#
# ML framing:
#   This is BINARY TEXT CLASSIFICATION.
#   Input  : raw message text
#   Output : one of two classes — spam (1) or ham (0)
#
# What does success look like?
#   - High RECALL on spam: we want to catch as much spam as
#     possible. Missing spam means it reaches the inbox.
#   - High PRECISION on spam: we don't want to misclassify
#     real messages as spam. A missed birthday invite because
#     it got flagged is a trust-breaking failure.
#   - These two goals are in tension — improving one often
#     hurts the other. We'll need to choose a balance (F1)
#     and inspect the tradeoff explicitly during evaluation.
#
# Constraints to keep in mind:
#   - Spam is the MINORITY class in real life (~10-15% of
#     messages). The model must not just lazily predict "ham"
#     every time to get a high accuracy score.
#   - Predictions need to be fast — this could run on every
#     incoming message in near real-time.
#   - The model needs to generalize to messages it's never
#     seen, not just memorize the training set.

print("=" * 55)
print("STAGE 1 — PROBLEM UNDERSTANDING")
print("=" * 55)
print("Task               : Binary text classification")
print("Classes            : spam (1), ham (0)")
print("Primary metric     : F1-score on spam class")
print("Secondary metrics  : Precision, Recall, ROC-AUC")
print("Known constraint   : Class imbalance (~13% spam)")
print("Success definition : Catch most spam (recall) without")
print("                     flagging too many real messages")
print("                     as spam (precision)")


# ────────────────────────────────────────────────────────────
# STAGE 2 — DATASET COLLECTION
# ────────────────────────────────────────────────────────────
#
# PLAIN ENGLISH:
# We use the SMS Spam Collection — a well-known, real public
# dataset of 5,572 real text messages, each manually labeled
# spam or ham by the original researchers. This isn't synthetic
# or generated data — it's actual messages people received.
#
# Why this dataset:
#   - Real-world labeled data, not synthetic
#   - Reasonably sized for a portfolio project (fast to train,
#     but big enough to be meaningful)
#   - Naturally imbalanced, which mirrors real spam filtering
#     and forces proper evaluation practices
#   - Widely used and well-documented, so results are easy to
#     sanity-check against published benchmarks
#
# Source: UCI Machine Learning Repository / SMS Spam Collection
# Format: tab-separated values, two columns: label, message

import pandas as pd

df = pd.read_csv("data/sms_spam_collection.tsv", sep="\t",
                  header=None, names=["label", "message"])

print()
print("=" * 55)
print("STAGE 2 — DATASET COLLECTION")
print("=" * 55)
print(f"Source       : SMS Spam Collection (UCI / public dataset)")
print(f"Total rows   : {len(df)}")
print(f"Columns      : {list(df.columns)}")
print()
print("Sample rows:")
print(df.head(5).to_string(index=False))
