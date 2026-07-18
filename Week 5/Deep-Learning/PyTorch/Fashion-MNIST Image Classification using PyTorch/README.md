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
