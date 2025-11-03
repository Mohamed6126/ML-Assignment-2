# 🧠 MNIST Classification — Logistic Regression, Softmax, Neural Network & CNN

This project implements and compares multiple machine learning models — **Logistic Regression**, **Softmax Regression**, a **Fully Connected Neural Network (MLP)**, and a **Convolutional Neural Network (CNN)** — trained on the **MNIST handwritten digits dataset**.

The goal is to explore how model architecture affects **accuracy**, **training time**, and **generalization**, progressing from simple linear classifiers to deep learning architectures.

---

## 📊 Project Overview

| Model | Type | Accuracy | Notes |
|:------|:------|:-----------|:------|
| Logistic Regression | Linear | ~98% | Fast but limited in feature representation |
| Softmax Regression | Multiclass Linear | ~90% | Handles multi-class efficiently |
| Fully Connected NN | 3 Hidden Layers | ~97% | Captures non-linear patterns |
| CNN | 2 Conv + 2 FC Layers | ~98.9% | Excels at spatial feature learning |

---

## 🚀 Features

- Implemented **from scratch using PyTorch**
- Supports:
  - Binary & Multiclass classification
  - CPU/GPU training
  - Training visualization (loss & accuracy plots)
  - Confusion matrix & classification report
- Compares:
  - **Accuracy**
  - **Training time**
  - **Overfitting & generalization**

---

## 🧩 Dataset

**MNIST** is a classic dataset of handwritten digits (0–9) consisting of:

- 60,000 training images  
- 10,000 test images  
- Each image: 28×28 grayscale pixels

Dataset is loaded using `torchvision.datasets.MNIST`.

---

## 🏗️ Model Implementations

### 1. Logistic Regression
- Implements a linear model for binary/multiclass classification.
- Trains using CrossEntropyLoss.
- Serves as a baseline to compare against deeper models.

### 2. Softmax Regression
- Extends logistic regression to handle 10-class classification.
- Demonstrates the power of vectorized linear models.

### 3. Neural Network (MLP)
- Multiple fully connected layers (ReLU activation).
- Demonstrates non-linear feature learning.


### 4. Convolutional Neural Network (CNN)
- Two convolutional layers + pooling + fully connected layers.
- Leverages **spatial feature learning** to achieve top accuracy.
- Includes **Dropout** for regularization and improved generalization.
---

## 📈 Results & Analysis

### 🔹 Model Comparison
- **CNN** achieves the best validation accuracy (~98.9%), showing the benefit of spatial feature extraction.  
- **Softmax Regression** performs nearly as well with simpler computation, ideal for lightweight scenarios.  
- **Fully Connected NN** generalizes well but may overfit without dropout.  
- **Logistic Regression** serves as a fast baseline but lacks expressiveness.

### 🔹 Overfitting & Generalization
Adding **Dropout layers** improved generalization by reducing overfitting, especially in the CNN model.

### 🔹 Limitations
- CNNs require more computational resources and tuning.
- Logistic models struggle with non-linear digit variations.
- Small models converge faster but plateau early in accuracy.

---

**You can see the full details (complexity analysis, hyperparameter tuning, running time, and visualizations) in the provided Jupyter notebook.**


## ⚙️ Installation & Usage

```bash
git clone https://github.com/Mohamed6126/ML-Assignment-2
cd ML-Assignment-2

pip install torch torchvision matplotlib numpy scikit-learn

python Neural_Network.ipynb
python logistic.ipynb
