# Speech Emotion Recognition (SER)

This repository contains three machine learning pipelines for detecting emotions from voice recording data. The models are trained on a combination of the **TESS** (Toronto Emotional Speech Set) and **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) speech datasets.

---

## Model Comparison

| Feature | Pipeline 1: SVM Baseline | Pipeline 2: 2D CNN | Pipeline 3: Hybrid CRNN with Attention (Deeper) |
| :--- | :--- | :--- | :--- |
| **Input Representation** | 1D Averaged acoustic features (MFCC, Chroma, Mel, Contrast) | 2D Mel-Spectrogram (128 Mel bands x 128 Time frames) | 2D Mel-Spectrogram (128 Mel bands x 128 Time frames) |
| **Dimensions** | 198-dimensional vector | 128 x 128 matrix | 128 x 128 matrix |
| **Model Type** | Support Vector Machine (SVM) with RBF kernel | 4-Layer Convolutional Neural Network (CNN) | Conv2D Feature Extractor + Bidirectional LSTM + Self-Attention |
| **Overall Accuracy** | **78.95%** | **79.39%** | **82.53%** (Best) |
| **Output Type** | Predicted Emotion | Predicted Emotion | Predicted Emotion + **Attention Timeline** |

---

## Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```
*(Note: If you do not have a GPU, it is recommended to install the CPU-only version of PyTorch).*

---

## Pipeline 1: SVM Baseline

### 1. Preprocess
Extract average 1D acoustic features:
```bash
python preprocess.py
```
This generates `extracted_features.csv`.

### 2. Train
Train the SVM model:
```bash
python train.py
```
This saves `emotion_svm_model.joblib`, `scaler.joblib`, and `classes.joblib`.

### 3. Predict
Run prediction on an audio file:
```bash
python predict.py --file "path/to/your/audio.wav"
```

---

## Pipeline 2: PyTorch 2D CNN

### 1. Preprocess
Extract 2D Mel-spectrograms of size 128x128:
```bash
python preprocess_cnn.py
```
This generates `extracted_features_cnn.npz`.

### 2. Train
Train the 2D CNN network:
```bash
python train_cnn.py
```
This saves the trained weights to `emotion_cnn_model.pth`, scaling params to `normalization_params.joblib`, and class labels to `classes_cnn.joblib`.

### 3. Predict
Run prediction on an audio file:
```bash
python predict_cnn.py --file "path/to/your/audio.wav"
```

---

## Pipeline 3: Hybrid CRNN with Attention (Best Performance)

### 1. Preprocess
Reuses the exact same preprocessed file from the CNN pipeline:
```bash
python preprocess_cnn.py
```
*(No need to run again if `extracted_features_cnn.npz` already exists).*

### 2. Train
Train the Hybrid CRNN + Attention network:
```bash
python train_crnn.py
```
This saves the trained weights to `emotion_crnn_model.pth`, scaling params to `normalization_params_crnn.joblib`, and class labels to `classes_crnn.joblib`.

### 3. Predict
Run prediction on an audio file (includes printing a timeline of where the model focused its attention in the audio):
```bash
python predict_crnn.py --file "path/to/your/audio.wav"
```

---

## Real-time Microphone Client

You can run the interactive real-time prediction script to record 3 seconds of your voice from your microphone and get immediate feedback:
```bash
python predict_realtime.py
```
*(Make sure to use the full Python path on your device: `C:\Users\Intel\AppData\Local\Programs\Python\Python313\python.exe predict_realtime.py`)*
