# Melanoma Binary Classifier 🧠

This project uses the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) to detect **malignant melanoma** (dangerous skin cancer) versus **benign skin lesions** using a simple Convolutional Neural Network (CNN) implemented in PyTorch.

---

## 📁 Dataset

- **Source**: HAM10000 (Kaggle)
- **Classes used**:
  - `mel`: malignant melanoma → label **1**
  - `nv`: benign nevus → label **0**
- **Total samples** used: 2-class subset from original dataset
- **Train/Validation/Test split**: 70% / 20% / 10%

---

## 🧠 Model Overview

- Model: Simple CNN with 3 convolutional layers and 2 fully connected layers
- Activation functions: ReLU + Sigmoid (for binary output)
- Output: binary prediction (0 = benign, 1 = melanoma)
- Loss: `BCEWithLogitsLoss` with class weighting to handle imbalance

