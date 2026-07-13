# 🧠 Week 4 – Deep Learning Foundations with NumPy

Welcome to **Week 4** of my AI learning journey.

This folder contains my **Deep Learning Foundations** implemented and explored using **Python** and **NumPy** before moving to professional frameworks like **PyTorch**.

The goal of this week was **not** to build production-ready neural networks, but to develop a deep understanding of the mathematics, intuition, and engineering concepts behind modern Deep Learning.

Instead of relying on high-level libraries, every lesson was studied from first principles so that I understand **what happens inside a neural network**, not just how to call library functions.

---

# 🎯 Learning Objectives

Throughout these lessons, I focused on understanding:

- Neural Networks from first principles
- Mathematical foundations of Deep Learning
- Matrix operations and vectorization
- Forward propagation
- Activation functions
- Loss functions
- Gradient Descent
- Backpropagation
- Optimization algorithms
- Regularization techniques
- Model evaluation
- Training workflow
- Hyperparameter tuning

This foundation prepares me for implementing Deep Learning models using **PyTorch** in the next stage of my learning journey.

---

# 📚 Lessons Covered

## 1. Deep Learning Basics

- Difference between Machine Learning and Deep Learning
- Biological neuron vs Artificial neuron
- Neural network intuition
- Real-world applications

---

## 2. NumPy for Deep Learning

- Arrays
- Matrix operations
- Dot products
- Broadcasting
- Why NumPy is the backbone of Deep Learning

---

## 3. Vectorization

- Why loops are slow
- Batch computation
- Matrix multiplication
- Efficient computation for neural networks

---

## 4. Artificial Neuron

Implemented and understood:

- Inputs
- Weights
- Bias
- Weighted sum

\[
z = w^Tx + b
\]

---

## 5. Activation Functions

Studied:

- Binary Step
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU

Compared:

- Output ranges
- Advantages
- Disadvantages
- Real-world usage

---

## 6. Dense (Fully Connected) Layers

Learned:

- Multiple neurons
- Hidden layers
- Network architecture
- Information flow

---

## 7. Forward Propagation

Implemented the complete forward pass:

Input

↓

Weighted Sum

↓

Activation

↓

Prediction

---

## 8. Loss Functions

Implemented:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Learned why loss functions guide learning.

---

## 9. Numerical Gradient

Used finite differences to estimate gradients.

Understood why derivatives matter in optimization.

---

## 10. Gradient Descent

Learned:

- Learning rate
- Weight updates
- Convergence
- Overshooting

---

## 11. Backpropagation

Studied:

- Chain Rule
- Error propagation
- Weight updates across layers

---

## 12. Epochs

Understood:

- Epoch
- Iteration
- Training cycles
- Why multiple epochs are needed

---

## 13. Batch Training

Compared:

- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent

---

## 14. Data Shuffling & DataLoader Concepts

Learned:

- Why data is shuffled
- Mini-batch generation
- Python generators (`yield`)
- Efficient data loading

---

## 15. Optimizers

Explored:

- Gradient Descent
- Momentum
- RMSProp
- Adam

Learned how modern optimizers improve training.

---

## 16. Overfitting & Regularization

Studied:

- Underfitting
- Good fit
- Overfitting

Techniques:

- Early Stopping
- L2 Regularization
- Collecting more data

---

## 17. Weight Initialization

Compared:

- Zero Initialization
- Random Initialization
- Xavier Initialization
- He Initialization

Learned why initialization affects convergence.

---

## 18. Batch Normalization

Implemented Batch Normalization using NumPy.

Learned:

- Mean normalization
- Standard deviation normalization
- Gamma (γ)
- Beta (β)

---

## 19. Dropout

Implemented Dropout.

Learned:

- Co-adaptation
- Random neuron deactivation
- Inverted Dropout
- Regularization

---

## 20. Learning Rate Scheduling

Explored:

- Step Decay
- Exponential Decay
- ReduceLROnPlateau
- Cosine Annealing (concept)

---

## 21. Evaluation Metrics

Implemented:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Learned why Accuracy alone is often misleading.

---

## 22. Model Checkpointing & Hyperparameter Tuning

Studied:

### Model Checkpointing

- Saving weights
- Loading weights
- Best model vs Last model

### Hyperparameter Tuning

Explored:

- Learning Rate
- Batch Size
- Hidden Layers
- Hidden Units
- Dropout Rate
- Optimizer Selection

---

# 🛠 Technologies Used

- Python
- NumPy
- VS Code
- Git
- GitHub

---

# 📂 Folder Structure

```text
Deep Learning/
│
├── 1-basics.py
├── 2-numpy.py
├── 3-vectorization.py
├── ...
├── 22-hyperparameter_tuning.py
└── README.md
```

---

# 🎓 Key Learning Outcomes

After completing this module, I can confidently explain:

- How a neural network works internally
- How forward propagation computes predictions
- How backpropagation updates weights
- Why activation functions are necessary
- How optimization algorithms improve learning
- Why regularization techniques reduce overfitting
- How Batch Normalization stabilizes training
- Why Dropout improves generalization
- How learning rate scheduling accelerates convergence
- How evaluation metrics should be selected based on the problem
- Why checkpointing is essential in real-world training pipelines
- How hyperparameters influence model performance

---

# 🚀 What's Next?

This marks the completion of my **Deep Learning Foundations**.

The next stage of my journey focuses on **PyTorch**, where I will apply these concepts using industry-standard tools to build professional Deep Learning projects.

Upcoming topics include:

- PyTorch Fundamentals
- Tensor Operations
- Automatic Differentiation (Autograd)
- Neural Networks with `nn.Module`
- Custom Training Loops
- GPU Acceleration
- CNNs (Convolutional Neural Networks)
- Real-world Deep Learning Projects

---

# 📌 Note

This repository is part of my long-term AI learning roadmap.

The purpose of these files is educational: to understand the core principles of Deep Learning before transitioning to production frameworks such as PyTorch.

Understanding the theory behind the tools allows me to build, debug, and explain Deep Learning systems with confidence.

---

## 👨‍💻 Author

**Mujahid Ayaz**

AI Engineer

Currently learning and building projects in:

- Machine Learning
- Deep Learning
- Computer Vision
- Natural Language Processing
- Generative AI