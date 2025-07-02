# 🫀 ECG Arrhythmia Detection using Deep Learning

This project focuses on detecting and classifying cardiac arrhythmias from ECG (Electrocardiogram) signal data using a 1D Convolutional Neural Network (CNN). The model is trained on preprocessed heartbeat segments from the MIT-BIH Arrhythmia Database.

---

## 📌 Objective

To develop a deep learning model that can **automatically classify ECG beats** into various types of arrhythmias — enabling fast, accurate, and automated diagnosis that can be integrated into real-time health monitoring systems.

---

## 📚 Dataset

The project uses the **MIT-BIH Arrhythmia Database**, which contains annotated ECG recordings sampled at 360 Hz. The dataset is loaded from the local path:


### Classes Considered:

* **N**: Normal
* **L**: Left bundle branch block beat
* **R**: Right bundle branch block beat
* **A**: Atrial premature beat
* **V**: Premature ventricular contraction

---

## 🛠️ Technologies and Libraries

* **Python**, **NumPy**, **Pandas**, **Matplotlib**
* **SciPy**, **PyWavelets**
* **Keras** (TensorFlow backend)
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**

---

## 🔄 Workflow Description

### 1. **Signal Extraction**

All files from the ECG database folder are read and parsed using a loop. Each signal is segmented based on annotations and stored in `X` and `y` lists for features and labels, respectively.

```python
filenames = next(os.walk(filepath))[2]
for f in filenames:
    ...
    signaldata = ...
    annotations = ...
```

---

### 2. **Preprocessing**

* **Wavelet Transform (`pywt`)** is applied to denoise the ECG signals.
* Each segment is **normalized and reshaped** to a fixed size (360 data points) to ensure consistency in CNN input.

---

### 3. **Handling Imbalance with SMOTE**

Due to the natural imbalance in arrhythmia classes, **Synthetic Minority Oversampling Technique (SMOTE)** is used to synthetically generate new samples for underrepresented classes.

```python
sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(X, y)
```

---

### 4. **Label Encoding**

Labels are converted into one-hot encoded vectors for classification using `LabelEncoder` and `to_categorical`.

---

### 5. **Model Architecture (1D CNN)**

A simple yet effective 1D CNN is implemented using Keras:

```python
model = Sequential()
model.add(Conv1D(...))
model.add(AvgPool1D(...))
model.add(Flatten())
model.add(Dense(..., activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
```

The model learns spatial dependencies and patterns in ECG waveforms to classify beats.

---

### 6. **Training the Model**

* Optimizer: `Adam`
* Loss Function: `categorical_crossentropy`
* Epochs: Defined based on dataset
* Validation set split using `train_test_split`

Training metrics (accuracy and loss) are plotted to track performance over time.

---

### 7. **Evaluation**

* The model is evaluated on a test set.
* A **confusion matrix** is plotted to visualize the classification results across arrhythmia types.
* Accuracy is printed and interpreted.

---

## 📊 Results and Observations

* The CNN model shows strong accuracy in detecting Normal and certain abnormal classes like `L` and `R`.
* SMOTE helped in balancing performance for minority classes (`A`, `V`).
* Visualization of training vs validation accuracy indicates whether the model is overfitting or underfitting.

---

## 🔮 Potential Improvements

* Use **Bidirectional LSTM** or **Transformer-based models** for sequential dependencies.
* Deploy the model via **Streamlit** for web-based ECG uploads.
* Train on larger datasets with more class granularity.

---

👥 Contributors
✨ Rithvika T
✨ Monish
✨ Manni
---

Let me know if you'd like this exported into a proper `README.md` file or if you'd like badges, preview images, or citations for datasets or academic work.
