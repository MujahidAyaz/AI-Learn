# ============================================================
#  STAGE 3 — EXPLORATORY DATA ANALYSIS (EDA)
#  Email/SMS Spam Classifier
# ============================================================
#
# PLAIN ENGLISH:
# Before touching any model, we look at the data with our own
# eyes. What does spam actually look like compared to normal
# messages? Are there obvious differences we can already see?
# EDA tells us what to expect and catches problems early —
# missing values, weird duplicates, class imbalance, etc.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/sms_spam_collection.tsv", sep="\t",
                  header=None, names=["label", "message"])

print("=" * 55)
print("STAGE 3 — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ── Basic shape and balance ──
print(f"Total messages   : {len(df)}")
print(f"Duplicate rows    : {df.duplicated().sum()}")
print(f"Missing values    : {df.isnull().sum().sum()}")
print()
print("Class balance:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3) * 100, "%")

# Real-world insight: spam is the minority class (~13%).
# This means accuracy alone will be misleading later —
# a model that always predicts "ham" would already be 87% accurate
# but completely useless. We'll need precision/recall/F1.

# ── Message length comparison ──
df["msg_length"] = df["message"].apply(len)
df["word_count"] = df["message"].apply(lambda x: len(x.split()))

print()
print("Average message length (characters):")
print(df.groupby("label")["msg_length"].mean().round(1))
print()
print("Average word count:")
print(df.groupby("label")["word_count"].mean().round(1))

# Real-world insight: spam messages tend to be noticeably
# longer than normal texts — they're trying to sell something
# or create urgency, which takes more words.

# ── Visualize ──
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("EDA — SMS Spam Dataset", fontsize=14, fontweight="bold")

# Class balance bar chart
counts = df["label"].value_counts()
axes[0,0].bar(counts.index, counts.values, color=["#4ade80", "#fb7185"])
axes[0,0].set_title("Class Balance (Ham vs Spam)")
axes[0,0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 50, str(v), ha="center", fontweight="bold")

# Message length distribution
axes[0,1].hist(df[df.label=="ham"]["msg_length"], bins=40, alpha=0.6,
               label="ham", color="#4ade80")
axes[0,1].hist(df[df.label=="spam"]["msg_length"], bins=40, alpha=0.6,
               label="spam", color="#fb7185")
axes[0,1].set_title("Message Length Distribution")
axes[0,1].set_xlabel("Characters")
axes[0,1].legend()

# Word count boxplot
df.boxplot(column="word_count", by="label", ax=axes[1,0],
           patch_artist=True,
           boxprops=dict(facecolor="#60a5fa"))
axes[1,0].set_title("Word Count by Class")
axes[1,0].set_xlabel("")
plt.suptitle("")  # remove auto title from boxplot

# Most common words in spam (simple split, no cleaning yet)
from collections import Counter
spam_words = " ".join(df[df.label=="spam"]["message"]).lower().split()
spam_words = [w.strip(".,!?:;") for w in spam_words if len(w) > 3]
top_spam = Counter(spam_words).most_common(10)
words, freqs = zip(*top_spam)
axes[1,1].barh(words[::-1], freqs[::-1], color="#fb7185")
axes[1,1].set_title("Top 10 Words in Spam (raw, uncleaned)")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("EDA chart saved to: eda_overview.png")
print()
print("Key takeaways feeding into next stages:")
print("  1. Class imbalance (13% spam) -> use stratified split + F1 score")
print("  2. Spam messages are longer -> message length could be a useful feature")
print("  3. Spam has distinct vocabulary -> TF-IDF will capture this well")
print("  4. No missing values -> cleaning will focus on text normalization")
