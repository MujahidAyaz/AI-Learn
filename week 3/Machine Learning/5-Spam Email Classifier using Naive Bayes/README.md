# 📧 Spam Email Classifier using Naive Bayes

## Overview

This project builds a Machine Learning model that classifies SMS messages as either:

- Spam
- Ham (Not Spam)

The model is trained using the Multinomial Naive Bayes algorithm and uses CountVectorizer to convert text into numerical features.

---

## Algorithm Used

- Multinomial Naive Bayes

---

## Dataset

SMS Spam Collection Dataset

Source:
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

---

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- CountVectorizer
- Multinomial Naive Bayes

---

## Workflow

1. Load dataset
2. Explore dataset
3. Handle duplicates
4. Convert labels into numbers
5. Separate Features and Target
6. Convert text into vectors
7. Split dataset
8. Train Naive Bayes model
9. Make predictions
10. Evaluate performance
11. Predict custom SMS messages

---

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

---

## Project Structure

```
Spam_Email_Classifier/
│
├── main.py
├── README.md
├── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Example Output

```
Accuracy: 98%

Prediction:
SPAM

Prediction:
HAM
```

---

## Skills Demonstrated

- Natural Language Processing (NLP)
- Text Vectorization
- Machine Learning
- Classification
- Data Preprocessing
- Model Evaluation
- Python Programming

---

## Future Improvements

- Save trained model using Joblib
- Build a Streamlit web application
- Deploy the model
- Try TF-IDF Vectorizer
- Compare with Logistic Regression
- Add user interface

---

Author

**Mujahid Ayaz**

Machine Learning & AI Learning Journey