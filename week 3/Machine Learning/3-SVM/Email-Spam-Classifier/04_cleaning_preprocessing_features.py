# ============================================================
#  STAGE 4-6 — CLEANING, PREPROCESSING, FEATURE ENGINEERING
#  Email/SMS Spam Classifier
# ============================================================
#
# PLAIN ENGLISH:
# Raw text is messy. "FREE!!!" and "free" mean the same thing
# to a human but look completely different to a computer.
# Cleaning normalizes all of this so the model can see the
# real patterns instead of getting confused by formatting noise.

import pandas as pd
import re
import string

df = pd.read_csv("data/sms_spam_collection.tsv", sep="\t",
                  header=None, names=["label", "message"])

print("=" * 55)
print("STAGE 4 — CLEANING")
print("=" * 55)

before = len(df)

# Remove exact duplicate messages — they'd otherwise let the
# model "cheat" by seeing the same message in train and test
df = df.drop_duplicates(subset="message").reset_index(drop=True)

print(f"Rows before dedup : {before}")
print(f"Rows after dedup  : {len(df)}")
print(f"Duplicates removed: {before - len(df)}")


# ────────────────────────────────────────────────────────────
# STAGE 5 — PREPROCESSING (text cleaning function)
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# We standardize every message the same way:
#   1. Lowercase everything   ("FREE" and "free" become identical)
#   2. Replace URLs with a placeholder token (the exact URL doesn't
#      matter — but "contains a URL" is a strong spam signal)
#   3. Replace long digit sequences with a placeholder
#      (phone numbers vary, but "contains a phone number" matters)
#   4. Strip punctuation and non-letter characters
#   5. Collapse extra whitespace

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\b\d{5,}\b", " PHONENUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Edge case: messages that are ONLY emoticons/numbers/symbols
    # (e.g. ":)" or "645") become an empty string after stripping
    # everything non-alphabetic. An empty string breaks TF-IDF
    # and gets misread as NaN when saved/reloaded from CSV.
    # We replace these rare edge cases with a placeholder token
    # instead of silently losing the row.
    if text == "":
        text = "emptymsg"
    return text

print()
print("=" * 55)
print("STAGE 5 — PREPROCESSING")
print("=" * 55)

df["clean_message"] = df["message"].apply(clean_text)

print("Before vs after cleaning (example):")
sample = df.iloc[2]
print(f"  Raw    : {sample['message']}")
print(f"  Clean  : {sample['clean_message']}")

# Encode labels: ham=0, spam=1 (models need numbers, not text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})


# ────────────────────────────────────────────────────────────
# STAGE 6 — FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────
# PLAIN ENGLISH:
# Beyond the words themselves, we can engineer extra signals
# that often differ a lot between spam and ham:
#   - message length (spam tends to be longer)
#   - number of capital letters (spam often SHOUTS)
#   - number of digits (phone numbers, prices, codes)
#   - number of exclamation marks (urgency tactics)
#   - number of special characters (£, $, %, *)
#
# These aren't fed into the text model directly here (TF-IDF
# handles the word patterns) but they're useful to inspect and
# could be added as extra numeric features in a more advanced
# version of this pipeline.

print()
print("=" * 55)
print("STAGE 6 — FEATURE ENGINEERING")
print("=" * 55)

df["msg_length"]    = df["message"].apply(len)
df["num_capitals"]  = df["message"].apply(lambda x: sum(1 for c in x if c.isupper()))
df["num_digits"]    = df["message"].apply(lambda x: sum(1 for c in x if c.isdigit()))
df["num_exclaim"]   = df["message"].apply(lambda x: x.count("!"))
df["num_special"]   = df["message"].apply(lambda x: sum(1 for c in x if c in string.punctuation))

engineered_summary = df.groupby("label")[
    ["msg_length", "num_capitals", "num_digits", "num_exclaim", "num_special"]
].mean().round(2)

print("Engineered feature averages by class:")
print(engineered_summary)

print()
print("Clear signal: spam messages average far more capitals,")
print("digits, and special characters than normal messages.")

# Save the cleaned dataset for the next stage (train/test split)
df.to_csv("data/cleaned_spam_data.csv", index=False)
print()
print("Cleaned dataset saved to: data/cleaned_spam_data.csv")
print(f"Final shape: {df.shape}")
