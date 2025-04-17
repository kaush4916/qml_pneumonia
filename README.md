# 🧠 Quantum Machine Learning for Pneumonia Detection

This project explores the integration of **Quantum Machine Learning (QML)** with classical deep learning techniques to detect **pneumonia from chest X-ray images**. We combine Convolutional Neural Networks (CNNs) with a quantum neural network layer using **Qiskit** and **PyTorch**, aiming to boost performance on medical image classification tasks.

---

## 📊 Dataset

We use the **Chest X-Ray Images (Pneumonia)** dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which contains:

- **Normal** and **Pneumonia** X-ray images
- Split into `train`, `val`, and `test` folders
- Grayscale images resized and normalized for input

---

## 🧠 Models & Techniques

Our hybrid model consists of:

- 🧱 **CNN Feature Extractor**: A custom CNN (based on ZFNet) to extract deep features from input images.
- ⚛️ **Quantum Neural Network (QNN)**: A parameterized quantum circuit (ZZFeatureMap + RealAmplitudes) wrapped with `TorchConnector` from Qiskit.
- 🔗 **Fusion Layer**: Classical CNN features are concatenated with QNN output and passed through fully connected layers.
- 🧪 **Training**: Performed using PyTorch on CPU, with early stopping and custom data loaders.

---

## 🚀 How to Run the Notebook

1. **Clone this repo:**

   ```bash
   git clone https://github.com/your-username/qml-pneumonia-detection.git
   cd qml-pneumonia-detection
