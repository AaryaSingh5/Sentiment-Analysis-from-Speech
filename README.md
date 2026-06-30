# Speech Emotion Recognition (SER)

This repository contains two machine learning pipelines for detecting emotions from voice recording data. The models are trained on a combination of the **TESS** (Toronto Emotional Speech Set) and **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) speech datasets.

---

## Model Comparison

| Feature | Pipeline 1: SVM Baseline | Pipeline 2: PyTorch 2D CNN (Deeper Model) |
| :--- | :--- | :--- |
| **Input Representation** | 1D Averaged acoustic features (MFCC, Chroma, Mel, Contrast) | 2D Mel-Spectrogram (128 Mel bands x 128 Time frames) |
| **Dimensions** | 198-dimensional vector | 128 x 128 matrix (preserving time structure) |
| **Model Type** | Support Vector Machine (SVM) with RBF kernel | 4-Layer Convolutional Neural Network (CNN) |
| **Overall Accuracy** | **78.95%** | **79.39%** |

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

## Pipeline 2: PyTorch 2D CNN (Deeper Model)

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
