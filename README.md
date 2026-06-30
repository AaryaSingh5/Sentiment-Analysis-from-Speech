# Speech Emotion Recognition (SER)

This repository contains a machine learning pipeline for detecting emotions from voice recording data. The model is trained on a combination of the **TESS** (Toronto Emotional Speech Set) and **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) speech datasets.

## Pipeline Architecture

The pipeline consists of:
1.  **Feature Extraction (`preprocess.py`)**: Extracts MFCCs, Chroma STFT, Mel-Spectrogram, and Spectral Contrast features from raw `.wav` audio files using `librosa`, creating a 198-dimensional acoustic vector.
2.  **Training (`train.py`)**: Standardizes the features and trains a Support Vector Machine (SVM) classifier with an RBF kernel. Saves the scaler, model, and class list.
3.  **Inference (`predict.py`)**: A lightweight script to predict the emotion of a new, single audio recording.

## Model Performance
*   **Overall Accuracy**: **78.95%**
*   **Emotions Detected**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Datasets
Make sure you have `archive (1)` and `archive (2)` datasets in your root directory. Run:
```bash
python preprocess.py
```
This generates `extracted_features.csv`.

### 2. Train the Model
Train the SVM model:
```bash
python train.py
```
This saves `emotion_svm_model.joblib`, `scaler.joblib`, and `classes.joblib`.

### 3. Predict Emotion
Predict the emotion of any `.wav` file:
```bash
python predict.py --file "path/to/your/audio.wav"
```
