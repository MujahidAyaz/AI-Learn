# 👕 Fashion-MNIST Image Classification using PyTorch

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

This project implements a complete **Deep Learning Image Classification Pipeline** using **PyTorch** to classify clothing images from the **Fashion-MNIST** dataset.

Instead of developing a single notebook, this project follows a **professional software engineering structure**, separating configuration, data loading, model architecture, training, evaluation, visualization, logging, checkpointing, and prediction into independent modules.

The objective is not only to train a neural network but also to demonstrate how a real-world Deep Learning project should be organized for maintainability, scalability, and reproducibility.

---

## 🎯 Problem Statement

The Fashion-MNIST dataset contains grayscale images of clothing items belonging to **10 different categories**.

The goal of this project is to train a neural network capable of correctly classifying unseen clothing images into their respective classes.

---

## 🧠 Dataset Information

- **Dataset:** Fashion-MNIST
- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Image Size:** 28 × 28 pixels
- **Color Channels:** 1 (Grayscale)
- **Number of Classes:** 10

### Classes

| Label | Class |
|------:|----------------|
| 0 | T-shirt / Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

# ✨ Features

✔ Modular Project Structure

✔ Professional Configuration Management

✔ Automatic Dataset Download

✔ DataLoader Pipeline

✔ Custom PyTorch Neural Network

✔ GPU / CPU Support

✔ Adam Optimizer

✔ Learning Rate Scheduler

✔ Early Stopping

✔ Model Checkpointing

✔ Automatic Logging

✔ Progress Bars (tqdm)

✔ Training History CSV Export

✔ Loss & Accuracy Curves

✔ Confusion Matrix

✔ Classification Report

✔ Prediction Visualization

✔ Model Summary (torchinfo)

✔ Professional Folder Structure

---

# 📂 Project Structure

```text
Fashion-MNIST Image Classification using PyTorch
│
├── configs/
│   └── config.py
│
├── data/
│
├── logs/
│
├── outputs/
│   ├── plots/
│   ├── predictions/
│   └── reports/
│
├── saved_models/
│
├── src/
│   ├── checkpoint.py
│   ├── dataset.py
│   ├── early_stopping.py
│   ├── engine.py
│   ├── history.py
│   ├── logger.py
│   ├── metrics.py
│   ├── model.py
│   ├── utils.py
│   └── visualization.py
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```
---

# 🏗️ Model Architecture

The classifier is implemented using a fully connected feed-forward neural network built with PyTorch.

```
Input Image (1 × 28 × 28)
            │
            ▼
Flatten Layer (784 Features)
            │
            ▼
Linear (784 → 128)
            │
            ▼
ReLU
            │
            ▼
Linear (128 → 64)
            │
            ▼
ReLU
            │
            ▼
Linear (64 → 10)
            │
            ▼
Class Probabilities
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Initial Learning Rate | 0.001 |
| LR Scheduler | ReduceLROnPlateau |
| Early Stopping | Enabled |
| Batch Size | 64 |
| Epochs | 10 |
| Device | CPU / GPU |
| Random Seed | 42 |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/MujahidAyaz/AI-Learn.git
```

Move into the project

```bash
cd "Week 5/Deep-Learning/PyTorch/Fashion-MNIST Image Classification using PyTorch"
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python train.py
```

During training the project automatically

- Downloads the dataset (if needed)
- Trains the neural network
- Saves the best model
- Generates training plots
- Saves training history
- Creates detailed log files

---

# 🔍 Evaluate the Model

```bash
python predict.py
```

This script generates

- Test Accuracy
- Classification Report
- Confusion Matrix
- Prediction Visualization

---

# 📊 Results

| Metric | Value |
|---------|------:|
| Best Validation Accuracy | **88.53%** |
| Dataset | Fashion-MNIST |
| Classes | 10 |
| Training Images | 60,000 |
| Testing Images | 10,000 |

---

# 📈 Generated Outputs

The project automatically generates:

```
outputs/
│
├── plots/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
│
├── predictions/
│   └── prediction_results.png
│
└── training_history.csv
```

---

# 📦 Saved Model

The best-performing model is automatically stored in

```
saved_models/
└── best_model.pth
```

---

# 📋 Logging

Every training run generates a timestamped log file.

Example:

```
logs/
└── training_2026-07-18_00-43-14.log
```

Each log contains:

- Training Progress
- Validation Metrics
- Learning Rate
- Checkpoint Information
- Final Accuracy

---

# 🎯 Project Highlights

✔ Professional Folder Structure

✔ Configuration-Based Design

✔ Modular Source Code

✔ Automatic Dataset Download

✔ Training History Export

✔ Learning Rate Scheduling

✔ Early Stopping

✔ Model Checkpointing

✔ Detailed Logging

✔ Progress Bars

✔ Model Evaluation

✔ Prediction Visualization

✔ Confusion Matrix

✔ Classification Report

✔ Ready for CNN Extension

---

# 🚀 Future Improvements

- Convolutional Neural Networks (CNN)
- Data Augmentation
- TensorBoard Integration
- ONNX Model Export
- Streamlit Web Application
- FastAPI REST API
- Docker Support
- CI/CD Pipeline
- Hyperparameter Optimization
- Transfer Learning

---

# 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- TorchInfo
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- tqdm

---

# 👨‍💻 Author

**Mujahid Ayaz**

Software Engineer | Machine Learning | Deep Learning | AI

GitHub:
https://github.com/MujahidAyaz

LinkedIn:
(www.linkedin.com/in/mujahid-ayaz)

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and supports future development.