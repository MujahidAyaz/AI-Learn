# Adult Income Classification using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge\&logo=python)

![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge\&logo=scikit-learn)

![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?style=for-the-badge)

![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</p>

---

# Overview

This project is a **production-style Machine Learning application** that predicts whether an individual's annual income exceeds **$50,000** based on demographic and employment-related attributes from the **Adult Income Dataset**.

Unlike notebook-based machine learning projects, this repository follows a **modular software engineering architecture** with separate components for configuration, data loading, preprocessing, model creation, training, evaluation, visualization, logging, and exception handling.

The objective of this project is not only to build accurate predictive models but also to demonstrate professional software engineering practices for machine learning systems.

---

# Project Objectives

The project aims to:

* Build an end-to-end supervised machine learning pipeline.
* Compare multiple classification algorithms under the same preprocessing workflow.
* Perform hyperparameter optimization using GridSearchCV.
* Evaluate models using multiple performance metrics.
* Automatically generate reports, visualizations, and trained model artifacts.
* Follow clean architecture principles suitable for production-quality machine learning projects.

---

# Features

✔ Automatic dataset download from the UCI Machine Learning Repository

✔ Local dataset caching

✔ Data validation

✔ Missing value handling

✔ Feature preprocessing using Scikit-learn Pipelines

✔ Separate preprocessing pipelines for linear and tree-based models

✔ One-Hot Encoding for categorical features

✔ Feature Scaling for distance-based algorithms

✔ Hyperparameter tuning with GridSearchCV

✔ Seven Machine Learning algorithms

✔ Automatic model serialization

✔ Comprehensive model evaluation

✔ ROC Curve generation

✔ Precision–Recall Curve generation

✔ Confusion Matrix visualization

✔ Feature Importance visualization

✔ Centralized configuration management

✔ Structured logging

✔ Custom exception handling

✔ Modular project architecture

---

# Machine Learning Models

The following algorithms are implemented and compared.

| Model                        | Hyperparameter Tuning |
| ---------------------------- | --------------------- |
| Logistic Regression          | ✅                     |
| K-Nearest Neighbors (KNN)    | ✅                     |
| Support Vector Machine (SVM) | Configurable          |
| Gaussian Naive Bayes         | ❌                     |
| Decision Tree                | ✅                     |
| Random Forest                | ✅                     |
| XGBoost                      | ✅                     |

---

# Tech Stack

### Programming Language

* Python 3.14

### Machine Learning

* Scikit-learn
* XGBoost

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Model Persistence

* Joblib

---

# Dataset

**Dataset Name**

Adult Income Dataset

**Source**

UCI Machine Learning Repository

**Prediction Target**

Income

* <=50K
* > 50K

**Number of Samples**

32,561

**Features**

14 input features

1 target column

The project automatically downloads the dataset during the first execution and stores a cached copy locally for future runs.

---

# Project Architecture

```text
                    +----------------+
                    |    main.py     |
                    +-------+--------+
                            |
                            v
                  +-------------------+
                  | Configure Logger  |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  |   Data Loader     |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  |  Preprocessing    |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Model Factory     |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Model Trainer     |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Model Evaluator   |
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | Visualization     |
                  +-------------------+
                            |
                            v
                  Reports / Models / Figures
```

---

# Project Structure

```text
FINAL-PROJECT-income-classification-comparison/

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│

├── reports/

│   ├── figures/

│   └── tables/

│

├── src/

│   ├── config.py

│   ├── logger.py

│   ├── exceptions.py

│   ├── data_loader.py

│   ├── preprocessing.py

│   ├── models.py

│   ├── trainer.py

│   ├── evaluator.py

│   └── visualization.py

│

├── logs/

│

├── main.py

├── requirements.txt

├── README.md

└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/MujahidAyaz/AI-Learn/week 4/Machine Learning/FINAL-PROJECT-income-classification-comparison.git
```

Move into the project directory

```bash
cd FINAL-PROJECT-income-classification-comparison
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install all required packages

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute the application using

```bash
python main.py
```

During execution the application will automatically:

1. Configure logging
2. Download or load the cached dataset
3. Validate the dataset
4. Preprocess the data
5. Train all machine learning models
6. Perform hyperparameter tuning where configured
7. Evaluate every trained model
8. Save trained models
9. Generate evaluation reports
10. Create visualizations

At the end of execution, trained models, comparison tables, logs, and publication-quality figures are automatically saved to their respective directories.

---

# Machine Learning Pipeline

The project follows a structured machine learning workflow designed to ensure reproducibility, maintainability, and separation of concerns.

## 1. Data Acquisition

The application automatically downloads the Adult Income dataset from the UCI Machine Learning Repository during the first execution. To avoid unnecessary network requests, the dataset is cached locally and reused in subsequent runs.

---

## 2. Data Validation

Before training begins, the dataset undergoes a validation process that checks:

* Dataset availability
* Expected number of columns
* Missing required columns
* Duplicate records
* Empty datasets
* Missing value symbols

This ensures that the training pipeline always receives valid input data.

---

## 3. Data Preprocessing

The preprocessing module prepares the dataset for machine learning by performing the following operations:

* Train/Test Split
* Target Encoding
* Missing Value Imputation
* Numerical Feature Scaling
* One-Hot Encoding of Categorical Variables
* Feature Type Identification

Two independent preprocessing pipelines are created:

### Scaled Pipeline

Used for algorithms sensitive to feature magnitude:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gaussian Naive Bayes

This pipeline includes:

* Median imputation for numerical features
* Most frequent imputation for categorical features
* StandardScaler
* OneHotEncoder

---

### Tree-Based Pipeline

Used for algorithms that do not require feature scaling:

* Decision Tree
* Random Forest
* XGBoost

This pipeline performs:

* Median imputation
* Most frequent categorical imputation
* OneHotEncoder

without applying feature scaling.

---

# Hyperparameter Optimization

Selected machine learning models are optimized using **GridSearchCV** with **5-fold cross-validation**.

The objective is to identify the best combination of hyperparameters while maintaining a fair comparison between algorithms.

Hyperparameter tuning improves model performance and reduces the likelihood of selecting suboptimal configurations.

---

# Model Evaluation

Instead of relying solely on Accuracy, multiple evaluation metrics are used to provide a comprehensive assessment of each classifier.

The project computes:

| Metric                                 | Description                                           |
| -------------------------------------- | ----------------------------------------------------- |
| Accuracy                               | Overall prediction accuracy                           |
| Balanced Accuracy                      | Accuracy adjusted for class imbalance                 |
| Precision                              | Positive prediction quality                           |
| Recall                                 | Ability to identify positive samples                  |
| F1 Score                               | Harmonic mean of Precision and Recall                 |
| ROC AUC                                | Area Under the ROC Curve                              |
| PR AUC                                 | Area Under the Precision–Recall Curve                 |
| Matthews Correlation Coefficient (MCC) | Balanced performance metric for binary classification |

These metrics allow objective comparison across different machine learning algorithms.

---

# Model Performance

The final evaluation results obtained on the test dataset are shown below.

| Rank | Model                  |   Accuracy |   F1 Score |    ROC AUC |        MCC |
| ---: | ---------------------- | ---------: | ---------: | ---------: | ---------: |
| 🥇 1 | XGBoost                | **87.67%** | **0.7241** | **0.9317** | **0.6486** |
| 🥈 2 | Random Forest          |     86.98% |     0.6989 |     0.9207 |     0.6237 |
| 🥉 3 | Decision Tree          |     85.77% |     0.6927 |     0.9050 |     0.6011 |
|    4 | Support Vector Machine |     85.97% |     0.6736 |     0.9056 |     0.5927 |
|    5 | Logistic Regression    |     85.58% |     0.6734 |     0.9078 |     0.5858 |
|    6 | K-Nearest Neighbors    |     84.28% |     0.6559 |     0.8846 |     0.5558 |
|    7 | Gaussian Naive Bayes   |     53.82% |     0.4983 |     0.7495 |     0.3294 |

---

# Best Performing Model

Among all evaluated algorithms, **XGBoost** achieved the strongest overall performance.

### Performance Summary

* Accuracy: **87.67%**
* F1 Score: **0.7241**
* ROC AUC: **0.9317**
* PR AUC: **0.8378**
* MCC: **0.6486**

These results demonstrate the effectiveness of gradient boosting for structured tabular data and justify its selection as the final model for this project.

---

# Generated Outputs

Running the project automatically produces several artifacts.

## Trained Models

```text
models/

├── logistic_regression.joblib
├── knn.joblib
├── svm.joblib
├── naive_bayes.joblib
├── decision_tree.joblib
├── random_forest.joblib
└── xgboost.joblib
```

---

## Evaluation Reports

```text
reports/

└── tables/

    └── model_comparison.csv
```

---

## Generated Figures

```text
reports/

└── figures/

    ├── accuracy.png
    ├── precision.png
    ├── recall.png
    ├── f1_score.png
    ├── roc_auc.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── confusion_matrices.png
    ├── random_forest_feature_importance.png
    └── xgboost_feature_importance.png
```

---

# Logging

Every stage of the pipeline is logged automatically.

Examples include:

* Dataset loading
* Dataset validation
* Model training
* Hyperparameter tuning
* Model evaluation
* Figure generation
* Model persistence

Logs are written both to the console and to a dedicated log file, making debugging and monitoring significantly easier.

---

# Reproducibility

To ensure reproducible results, the project uses:

* Fixed random seeds
* Centralized configuration
* Deterministic preprocessing
* Cached datasets
* Saved trained models
* Modular pipeline design

Running the project multiple times under the same configuration produces consistent outputs.


# Engineering Decisions

Several design decisions were intentionally made during the development of this project to improve maintainability, scalability, and code quality.

## Modular Architecture

Instead of implementing the entire workflow in a single notebook or script, the project is divided into independent modules with clearly defined responsibilities.

Each module is responsible for a single concern, making the project easier to maintain, test, and extend.

---

## Configuration-Driven Design

Project constants, directory paths, dataset information, model parameters, and other configurable values are centralized within a dedicated configuration module.

This approach eliminates hard-coded values and simplifies future modifications.

---

## Pipeline-Based Preprocessing

All preprocessing operations are implemented using Scikit-learn Pipelines and ColumnTransformers.

This ensures:

* Prevention of data leakage
* Consistent preprocessing during training and inference
* Simplified integration with GridSearchCV
* Reusable preprocessing workflows

---

## Custom Exception Handling

Project-specific exception classes improve debugging and provide more meaningful error messages compared to generic Python exceptions.

---

## Centralized Logging

A dedicated logging module records every major step of the machine learning workflow.

Logging includes:

* Dataset operations
* Training progress
* Hyperparameter optimization
* Model persistence
* Evaluation
* Visualization generation

This greatly simplifies debugging and project monitoring.

---

## Model Persistence

Every trained model is automatically serialized using Joblib.

This enables:

* Model reuse without retraining
* Faster experimentation
* Simplified deployment

---

# Skills Demonstrated

This project demonstrates practical experience with the following technologies and concepts.

## Machine Learning

* Supervised Learning
* Binary Classification
* Model Comparison
* Hyperparameter Optimization
* Cross Validation
* Feature Engineering
* Performance Evaluation
* Model Selection

---

## Python

* Object-Oriented Programming
* Modular Programming
* Type Hinting
* Dataclasses
* Exception Handling
* Logging
* File Management

---

## Scikit-learn

* Pipelines
* ColumnTransformer
* GridSearchCV
* Cross Validation
* Metrics
* Preprocessing
* Model Persistence

---

## Software Engineering

* Clean Project Structure
* Separation of Concerns
* Configuration Management
* Reproducibility
* Maintainable Code
* Professional Documentation

---

# Key Learnings

Developing this project provided valuable experience in building production-style machine learning systems.

Some of the most important lessons include:

* Building modular ML applications instead of notebook-only solutions.
* Designing preprocessing pipelines that prevent data leakage.
* Comparing multiple algorithms under identical preprocessing conditions.
* Selecting evaluation metrics appropriate for imbalanced datasets.
* Organizing machine learning code using software engineering best practices.
* Writing maintainable, reusable, and scalable Python code.

---

# Future Improvements

Potential future enhancements include:

* Bayesian Hyperparameter Optimization
* Optuna Integration
* SHAP Explainability
* LIME Explanations
* Feature Selection Techniques
* MLflow Experiment Tracking
* Docker Containerization
* FastAPI Model Deployment
* Streamlit Interactive Dashboard
* CI/CD using GitHub Actions
* Automated Unit Testing
* Model Monitoring and Drift Detection

---

# Requirements

The project depends on the following major libraries:

* Python 3.14
* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib
* joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this project for educational and personal purposes.

---

# Author

**Mujahid Ayaz**

Machine Learning & Artificial Intelligence Enthusiast

GitHub: **https://github.com/MujahidAyaz**

LinkedIn: **https://linkedin.com/in/mujahid-ayaz**

---

# Acknowledgements

This project uses the **Adult Income Dataset** provided by the **UCI Machine Learning Repository**.

Special thanks to the open-source community and the developers of:

* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib

whose tools make practical machine learning development accessible to everyone.

---

# Repository Highlights

✔ End-to-End Machine Learning Pipeline

✔ Production-Style Project Structure

✔ Modular Architecture

✔ Hyperparameter Optimization

✔ Comprehensive Model Evaluation

✔ Automatic Report Generation

✔ Publication-Quality Visualizations

✔ Model Serialization

✔ Professional Logging

✔ Custom Exception Handling

✔ Reproducible Results

✔ GitHub Portfolio Ready

---

> **If you find this project useful or interesting, consider giving it a ⭐ on GitHub.**
