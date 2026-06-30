# ============================================================
#  STAGE 11 — PREDICTION PIPELINE
#  Email/SMS Spam Classifier
# ============================================================
#
# PLAIN ENGLISH:
# This is the file someone actually uses in production. It does
# NOT retrain anything — it loads the already-trained model and
# vectorizer from disk and uses them to classify brand new
# messages instantly. This is what would sit behind an API
# endpoint, a Chrome extension, or an email server filter.

import re
import joblib

# ────────────────────────────────────────────────────────────
# Load the saved model + vectorizer (same ones from Stage 10)
# ────────────────────────────────────────────────────────────
# Both files MUST come from the same training run. Using a
# vectorizer that wasn't fit on the same vocabulary as the
# model was trained with will silently produce garbage results.

model = joblib.load("models/svm_spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Model and vectorizer loaded successfully.")
print(f"Model type      : {type(model).__name__}")
print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")


# ────────────────────────────────────────────────────────────
# Same cleaning function used during training
# ────────────────────────────────────────────────────────────
# CRITICAL: a new message must go through the EXACT same
# cleaning steps as training data did. If cleaning differs even
# slightly, the model sees patterns it was never trained on.

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\b\d{5,}\b", " PHONENUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text == "":
        text = "emptymsg"
    return text


def predict_message(message, return_details=False):
    """
    Classify a single message as spam or ham.

    Parameters
    ----------
    message : str
        The raw, unprocessed message text.
    return_details : bool
        If True, also return the cleaned text and probability.

    Returns
    -------
    dict with keys: label, confidence, (cleaned_text if requested)
    """
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0][1]   # probability of being spam

    result = {
        "label": "SPAM" if prediction == 1 else "HAM",
        "confidence": round(float(probability), 4)
    }

    if return_details:
        result["cleaned_text"] = cleaned

    return result


def predict_batch(messages):
    """Classify a list of messages at once — more efficient than
    calling predict_message() in a loop for large batches."""
    cleaned = [clean_text(m) for m in messages]
    vecs = vectorizer.transform(cleaned)
    predictions = model.predict(vecs)
    probabilities = model.predict_proba(vecs)[:, 1]

    return [
        {"message": m, "label": "SPAM" if p == 1 else "HAM",
         "confidence": round(float(prob), 4)}
        for m, p, prob in zip(messages, predictions, probabilities)
    ]


# ────────────────────────────────────────────────────────────
# Demo — test on brand new messages the model has never seen
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("STAGE 11 — PREDICTION PIPELINE DEMO")
    print("=" * 60)

    test_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: Your account has been suspended. Verify immediately at this link.",
        "Mom said dinner is ready, come downstairs",
        "FREE entry in our weekly competition! Text WIN to 80082 now!!!",
        "Can you send me the report before end of day?",
        "You have been selected for a cash prize of $5000. Reply YES to claim.",
        "Running 10 mins late, see you soon",
    ]

    results = predict_batch(test_messages)

    print(f"{'Prediction':<8} {'Confidence':>11}   Message")
    print("-" * 75)
    for r in results:
        flag = "⚠" if r["label"] == "SPAM" else "✓"
        print(f"{flag} {r['label']:<6} {r['confidence']:>10.1%}   {r['message'][:55]}")

    print()
    print("Single message example with full details:")
    detailed = predict_message(test_messages[0], return_details=True)
    for k, v in detailed.items():
        print(f"  {k}: {v}")
