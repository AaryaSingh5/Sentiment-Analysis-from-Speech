import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Define directories
TESS_DIR = os.path.join("archive (1)", "TESS Toronto emotional speech set data")
CREMA_DIR = os.path.join("archive (2)", "Crema")
OUTPUT_CSV = "extracted_features.csv"

# Emotion mappings
CREMA_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

def extract_features(file_path):
    """
    Extracts acoustic features from an audio file:
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma STFT
    - Mel Spectrogram
    - Spectral Contrast
    """
    try:
        # Load audio file (sr=None preserves native sampling rate)
        y, sr = librosa.load(file_path, sr=None)
        
        # Check for empty files
        if len(y) == 0:
            return None
            
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
        
        # Combine all features into one vector (dimension: 80 + 24 + 80 + 14 = 198)
        feature_vector = np.hstack([
            mfccs_mean, mfccs_std,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std
        ])
        
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_datasets():
    data = []
    
    # --- 1. Process CREMA-D dataset ---
    if os.path.exists(CREMA_DIR):
        print("Processing CREMA-D dataset...")
        crema_files = glob.glob(os.path.join(CREMA_DIR, "*.wav"))
        for file_path in tqdm(crema_files):
            filename = os.path.basename(file_path)
            # Example filename: 1001_DFA_ANG_XX.wav
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = CREMA_MAP.get(emotion_code)
                if emotion:
                    features = extract_features(file_path)
                    if features is not None:
                        data.append({
                            'file': file_path,
                            'dataset': 'CREMA-D',
                            'label': emotion,
                            'features': features
                        })
    else:
        print(f"Warning: CREMA-D directory not found at {CREMA_DIR}")

    # --- 2. Process TESS dataset ---
    if os.path.exists(TESS_DIR):
        print("Processing TESS dataset...")
        # Find all wav files recursively under TESS directory
        tess_files = glob.glob(os.path.join(TESS_DIR, "**", "*.wav"), recursive=True)
        for file_path in tqdm(tess_files):
            # Extract parent folder name
            folder_name = os.path.basename(os.path.dirname(file_path))
            folder_name_lower = folder_name.lower()
            
            # Map folder name to emotion
            emotion = None
            if 'angry' in folder_name_lower:
                emotion = 'angry'
            elif 'disgust' in folder_name_lower:
                emotion = 'disgust'
            elif 'fear' in folder_name_lower:
                emotion = 'fear'
            elif 'happy' in folder_name_lower:
                emotion = 'happy'
            elif 'sad' in folder_name_lower:
                emotion = 'sad'
            elif 'neutral' in folder_name_lower:
                emotion = 'neutral'
            elif 'surprise' in folder_name_lower:
                emotion = 'surprise'
                
            if emotion:
                features = extract_features(file_path)
                if features is not None:
                    data.append({
                        'file': file_path,
                        'dataset': 'TESS',
                        'label': emotion,
                        'features': features
                    })
    else:
        print(f"Warning: TESS directory not found at {TESS_DIR}")

    if not data:
        print("Error: No data processed!")
        return

    # Convert to DataFrame
    print("Saving features to CSV...")
    feature_cols = [f'feat_{i}' for i in range(len(data[0]['features']))]
    
    rows = []
    for item in data:
        row = {
            'file': item['file'],
            'dataset': item['dataset'],
            'label': item['label']
        }
        for i, val in enumerate(item['features']):
            row[f'feat_{i}'] = val
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Preprocessing completed! Extracted features for {len(df)} files and saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_datasets()
