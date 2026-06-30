import os
import argparse
import numpy as np
import librosa
import joblib

# Paths to models
MODEL_PATH = "emotion_svm_model.joblib"
SCALER_PATH = "scaler.joblib"
CLASSES_PATH = "classes.joblib"

def extract_features(file_path):
    """
    Extracts acoustic features from an audio file:
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma STFT
    - Mel Spectrogram
    - Spectral Contrast
    Must match the preprocessing feature extraction logic exactly.
    """
    try:
        # Load audio file (sr=None preserves native sampling rate)
        y, sr = librosa.load(file_path, sr=None)
        
        if len(y) == 0:
            raise ValueError("Audio file is empty.")
            
        # 1. MFCC (40 coefficients) -> Mean and Std
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Compute STFT for chroma and spectral contrast
        stft = np.abs(librosa.stft(y))
        
        # 2. Chroma STFT (12 bins) -> Mean and Std
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # 3. Mel Spectrogram (40 mels) -> Mean and Std
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        # 4. Spectral Contrast (7 bands) -> Mean and Std
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        
        # Combine all features into one vector (dimension: 198)
        feature_vector = np.hstack([
            mfccs_mean, mfccs_std,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std
        ])
        
        return feature_vector
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a speech recording file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the audio WAV file.")
    args = parser.parse_args()

    # Verify model files exist
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(CLASSES_PATH)):
        print("Error: Model files not found! Please run train.py first to train and save the model.")
        return

    # Verify input file exists
    if not os.path.exists(args.file):
        print(f"Error: Input file '{args.file}' does not exist.")
        return

    print(f"Loading model components...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = joblib.load(CLASSES_PATH)

    print(f"Extracting features from {args.file}...")
    features = extract_features(args.file)
    if features is None:
        print("Failed to process the audio file.")
        return

    # Reshape features to 2D array for scaler/model (1 sample, N features)
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)

    # Predict emotion class and probabilities
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # Display results
    print("\n" + "="*50)
    print(f"Predicted Emotion: {prediction.upper()}")
    print("="*50)
    print("Confidence Scores:")
    for class_name, prob in zip(classes, probabilities):
        print(f" - {class_name.capitalize():<18}: {prob * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
