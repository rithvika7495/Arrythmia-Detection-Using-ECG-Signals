# 🫀 Arrythmia Detection Using ECG Signals

This project focuses on detecting and classifying cardiac arrhythmias from ECG (Electrocardiogram) signal data using a 1D Convolutional Neural Network (CNN). The model is trained on preprocessed heartbeat segments from the MIT-BIH Arrhythmia Database.

---

## 📌 Objective

To develop a deep learning model that can **automatically classify ECG beats** into various types of arrhythmias — enabling fast, accurate, and automated diagnosis that can be integrated into real-time health monitoring systems.

---

## 📚 Dataset

We use the **MIT-BIH Arrhythmia Database**, which contains annotated ECG recordings sampled at 360 Hz.

**Classes considered**:

* **N**: Normal
* **L**: Left bundle branch block beat
* **R**: Right bundle branch block beat
* **A**: Atrial premature beat
* **V**: Premature ventricular contraction

Each beat is segmented, labeled, and preprocessed for model training.

---

## 🛠️ Technologies and Libraries

* **Python**
* **NumPy**, **Pandas**, **Matplotlib**
* **SciPy**, **PyWavelets**
* **Scikit-learn**
* **Keras / TensorFlow**
* **Imbalanced-learn (SMOTE)**

---

## 🔄 Workflow Overview

### 1. **Data Extraction and Labeling**

* ECG signal files are read from the directory.
* Annotations are used to segment each signal into labeled heartbeats.
* Beat segments are stored in `X` and corresponding labels in `y`.

### 2. **Preprocessing**

* **Wavelet transforms** (`pywt`) are applied for denoising.
* All beats are **normalized and resized** to 360 data points.
* Classes are **encoded** using `LabelEncoder` and one-hot encoded.

### 3. **Class Imbalance Handling**

* **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance underrepresented classes like `A` and `V`.

### 4. **Model Architecture**

A **1D CNN** model is constructed using Keras:

```python
Sequential([
    Conv1D(...),
    AvgPool1D(...),
    Flatten(),
    Dense(...),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])
```

* The CNN captures local temporal features in ECG signals.

### 5. **Training and Evaluation**

* The dataset is split into training and test sets.
* The model is compiled using the **Adam optimizer** and **categorical crossentropy** loss.
* Training and validation accuracy/loss are plotted.
* A **confusion matrix** visualizes class-wise performance.

---

## 📊 Results

* The model performed well on majority classes and showed improved balance with SMOTE.
* Visualization tools like accuracy/loss curves and confusion matrices help assess performance.
* This model provides a strong baseline for future real-time arrhythmia detection systems.

---

## 👥 Contributors

* ✨ **Rithvika T**
* ✨ **Monish P**
* ✨ **Manni Chellappan Ramu**

