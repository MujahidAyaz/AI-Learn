"""
===========================================================
Project: Spam Email Classifier using Naive Bayes
Author: Mujahid Ayaz
Algorithm: Multinomial Naive Bayes

Dataset:
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

Description:
This project classifies SMS messages as Spam or Ham (Not Spam)
using the Multinomial Naive Bayes algorithm.

===========================================================
"""

# ==========================================================
# STEP 1 - Import Required Libraries
# ==========================================================

# Used for data manipulation
import pandas as pd

# Converts text into numerical word counts
from sklearn.feature_extraction.text import CountVectorizer

# Splits the dataset into training and testing data
from sklearn.model_selection import train_test_split

# Naive Bayes Algorithm
from sklearn.naive_bayes import MultinomialNB

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# STEP 2 - Load Dataset
# ==========================================================

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_table(
    url,
    header=None,
    names=["label", "message"],
)

print("=" * 60)
print("First Five Rows")
print(df.head())

# ==========================================================
# STEP 3 - Explore Dataset
# ==========================================================

print("=" * 60)
print("Dataset Shape")
print(df.shape)

print("=" * 60)
print("Dataset Information")
print(df.info())

print("=" * 60)
print("Missing Values")
print(df.isnull().sum())

print("=" * 60)
print("Duplicate Rows")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("=" * 60)
print("Shape After Removing Duplicates")
print(df.shape)

print("=" * 60)
print("Spam vs Ham")
print(df["label"].value_counts())

# ==========================================================
# STEP 4 - Convert Labels into Numbers
# ==========================================================

# Ham = 0
# Spam = 1

df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1,
    }
)

print("=" * 60)
print("Updated Dataset")
print(df.head())

# ==========================================================
# STEP 5 - Separate Features and Target
# ==========================================================

# X contains SMS messages

X = df["message"]

# y contains labels

y = df["label"]

# ==========================================================
# STEP 6 - Convert Text into Numerical Features
# ==========================================================

"""
Machine Learning algorithms cannot understand text.

CountVectorizer converts every SMS into numbers.

Example:

"Free money now"

becomes something similar to

free  money  now

1      1      1

"""

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

print("=" * 60)
print("Feature Matrix Shape")
print(X.shape)

# ==========================================================
# STEP 7 - Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("=" * 60)
print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==========================================================
# STEP 8 - Create Naive Bayes Model
# ==========================================================

model = MultinomialNB()

# ==========================================================
# STEP 9 - Train Model
# ==========================================================

model.fit(X_train, y_train)

print("=" * 60)
print("Model Training Completed")

# ==========================================================
# STEP 10 - Make Predictions
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# STEP 11 - Evaluate Model
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("Accuracy")
print(f"{accuracy:.4f}")

print("=" * 60)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("=" * 60)
print("Classification Report")
print(classification_report(y_test, y_pred))

# ==========================================================
# STEP 12 - Test Your Own Message
# ==========================================================

print("=" * 60)
print("Custom Prediction")

sample_message = [
    "Congratulations! You have won a FREE iPhone. Click here to claim now."
]

sample_vector = vectorizer.transform(sample_message)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction : SPAM")
else:
    print("Prediction : HAM")

# ==========================================================
# STEP 13 - Test Another Message
# ==========================================================

sample_message = [
    "Hey Ali, let's meet tomorrow evening for dinner."
]

sample_vector = vectorizer.transform(sample_message)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction : SPAM")
else:
    print("Prediction : HAM")

print("=" * 60)
print("Project Finished Successfully")
print("=" * 60)