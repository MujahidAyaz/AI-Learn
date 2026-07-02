# 🧠 Principal Component Analysis (PCA) using Scikit-Learn

## 📌 Project Overview

This project demonstrates how **Principal Component Analysis (PCA)** can reduce the dimensionality of a dataset while preserving most of its important information.

The project uses the **Breast Cancer Wisconsin Dataset** from Scikit-Learn and compares the performance of a **Logistic Regression** model before and after applying PCA.

This project follows a complete machine learning workflow including data preprocessing, feature scaling, dimensionality reduction, model training, evaluation, and visualization.

---

# 🎯 Objectives

- Understand Principal Component Analysis (PCA)
- Reduce the number of features while retaining important information
- Visualize high-dimensional data in 2D
- Compare model performance before and after PCA
- Learn when and why PCA should be used

---

# 📂 Dataset

**Dataset Name**

Breast Cancer Wisconsin Dataset

**Source**

Built into the Scikit-Learn library.

It contains measurements computed from breast cancer cell images and is commonly used for binary classification tasks.

### Dataset Information

- Total Samples: **569**
- Features: **30**
- Target Classes: **2**

| Target | Meaning |
|---------|---------|
| 0 | Malignant (Cancer) |
| 1 | Benign (Non-Cancer) |

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

# 📚 Machine Learning Concepts Covered

- Data Exploration
- Feature Scaling
- StandardScaler
- Principal Component Analysis (PCA)
- Explained Variance
- Cumulative Explained Variance
- Logistic Regression
- Model Evaluation
- Confusion Matrix
- Classification Report
- Data Visualization

---

# 📁 Project Structure

```
PCA/
│
├── Breast_Cancer_PCA.py
├── README.md
├── requirements.txt
└── 
```

---

# ⚙ Workflow

```
Load Dataset
        │
        ▼
Explore Dataset
        │
        ▼
Check Missing Values
        │
        ▼
Split Dataset
        │
        ▼
Feature Scaling
        │
        ▼
Train Logistic Regression (Without PCA)
        │
        ▼
Apply PCA
        │
        ▼
Analyze Explained Variance
        │
        ▼
Select Principal Components
        │
        ▼
Train Logistic Regression (With PCA)
        │
        ▼
Evaluate Model
        │
        ▼
Visualize Principal Components
```

---

# 📊 Principal Component Analysis (PCA)

PCA is an **unsupervised dimensionality reduction technique** that transforms a large number of correlated features into a smaller number of uncorrelated features called **Principal Components**.

Its primary goal is to retain as much information (variance) as possible while reducing the number of features.

---

# ❓ Why PCA?

High-dimensional datasets often contain:

- Redundant Features
- Highly Correlated Features
- Increased Computational Cost
- Higher Risk of Overfitting

PCA helps by:

- Reducing dimensionality
- Improving computational efficiency
- Simplifying visualization
- Removing redundant information

---

# 📈 Explained Variance

Each principal component explains a portion of the total variance in the dataset.

Example:

| Component | Explained Variance |
|-----------|-------------------:|
| PC1 | 44.2% |
| PC2 | 19.0% |
| PC3 | 9.4% |

The cumulative explained variance helps determine how many principal components should be retained.

---

# 🤖 Model Used

**Logistic Regression**

The project compares:

- Logistic Regression without PCA
- Logistic Regression after PCA

This comparison demonstrates the effect of dimensionality reduction on model performance.

---

# 📉 Evaluation Metrics

The trained model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

# 📊 Visualizations

The project generates:

- Cumulative Explained Variance Plot
- 2D PCA Scatter Plot

These visualizations help understand how much information is preserved and how the transformed data is distributed.

---

# 🚀 Installation

Clone the repository



Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python Breast_Cancer_PCA.py
```

---

# 📦 Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

---

# 📚 What You Will Learn

By completing this project, you will understand:

- What PCA is
- Why feature scaling is necessary before PCA
- How PCA reduces dimensionality
- How to interpret explained variance
- How to select the number of principal components
- How PCA affects machine learning performance
- How to visualize high-dimensional datasets

---

# 🔮 Future Improvements

- Apply PCA to larger datasets
- Compare PCA with t-SNE and UMAP
- Build an interactive Streamlit application
- Experiment with different classifiers after PCA
- Perform hyperparameter tuning
- Deploy the trained model

---

# 📖 References

- Scikit-Learn Documentation
- Breast Cancer Wisconsin Dataset
- Principal Component Analysis (PCA)

---

# 👨‍💻 Author

**Mujahid Ayaz**

Machine Learning & AI Learning Journey

GitHub Portfolio Project