import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# Define directories
TESS_DIR = os.path.join("archive (1)", "TESS Toronto emotional speech set data")
CREMA_DIR = os.path.join("archive (2)", "Crema")
OUTPUT_NPZ = "extracted_features_cnn.npz"

# Emotion mappings
CREMA_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Fixed parameters for 2D representation
N_MELS = 128
MAX_FRAMES = 128  # hop_length=512, sr=22050 -> 128 frames is ~3 seconds of audio

def extract_mel_spectrogram(file_path):
    """
    Extracts a 2D Mel-spectrogram from an audio file,
    resizing it to shape (128, 128) via padding or center cropping.
    """
    try:
        # Load audio (sr=22050 standardizes the sample rate)
        y, sr = librosa.load(file_path, sr=22050)
        
        if len(y) == 0:
            return None
            
        # Compute Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=512)
        # Convert power to decibels (log scale)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Handle time axis (axis 1) padding or cropping to hit MAX_FRAMES
        num_frames = mel_db.shape[1]
        
        if num_frames < MAX_FRAMES:
            # Pad on the right with the minimum value (silence in dB scale)
            pad_width = MAX_FRAMES - num_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_db.min())
        elif num_frames > MAX_FRAMES:
            # Center crop
            start_frame = (num_frames - MAX_FRAMES) // 2
            mel_db = mel_db[:, start_frame : start_frame + MAX_FRAMES]
            
        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_datasets():
    raw_data = []
    
    # 1. Process CREMA-D
    if os.path.exists(CREMA_DIR):
        print("Scanning CREMA-D dataset...")
        crema_files = glob.glob(os.path.join(CREMA_DIR, "*.wav"))
        for file_path in tqdm(crema_files):
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = CREMA_MAP.get(emotion_code)
                if emotion:
                    raw_data.append((file_path, emotion))
    else:
        print(f"Warning: CREMA-D directory not found at {CREMA_DIR}")

    # 2. Process TESS
    if os.path.exists(TESS_DIR):
        print("Scanning TESS dataset...")
        tess_files = glob.glob(os.path.join(TESS_DIR, "**", "*.wav"), recursive=True)
        for file_path in tqdm(tess_files):
            folder_name = os.path.basename(os.path.dirname(file_path))
            folder_name_lower = folder_name.lower()
            
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
                raw_data.append((file_path, emotion))
    else:
        print(f"Warning: TESS directory not found at {TESS_DIR}")

    if not raw_data:
        print("Error: No audio files found!")
        return

    # Extract unique classes
    classes = sorted(list(set([item[1] for item in raw_data])))
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(classes)} classes: {classes}")

    # Extract features
    print("Extracting 2D Mel-spectrograms...")
    X = []
    y = []
    
    for file_path, emotion in tqdm(raw_data):
        features = extract_mel_spectrogram(file_path)
        if features is not None:
            X.append(features)
            y.append(class_to_id[emotion])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Save to compressed file
    print(f"Saving dataset to {OUTPUT_NPZ}...")
    np.savez_compressed(OUTPUT_NPZ, X=X, y=y, classes=np.array(classes))
    print(f"Done! Saved dataset with shape {X.shape} and {len(classes)} classes.")

if __name__ == "__main__":
    process_datasets()
