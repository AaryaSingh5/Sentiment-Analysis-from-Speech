import os
import argparse
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn

# Paths to models
MODEL_PATH = "emotion_cnn_model.pth"
NORMALIZATION_PATH = "normalization_params.joblib"
CLASSES_PATH = "classes_cnn.joblib"

# Define 2D CNN Architecture (Must match training exactly)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def extract_mel_spectrogram(file_path):
    """
    Extracts a 2D Mel-spectrogram from an audio file,
    resizing it to shape (128, 128) via padding or center cropping.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050)
        if len(y) == 0:
            raise ValueError("Audio file is empty.")
            
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        num_frames = mel_db.shape[1]
        
        if num_frames < 128:
            pad_width = 128 - num_frames
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_db.min())
        elif num_frames > 128:
            start_frame = (num_frames - 128) // 2
            mel_db = mel_db[:, start_frame : start_frame + 128]
            
        return mel_db
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a speech WAV file using PyTorch 2D CNN.")
    parser.add_argument("--file", type=str, required=True, help="Path to the audio WAV file.")
    args = parser.parse_args()

    # Check model files
    if not (os.path.exists(MODEL_PATH) and os.path.exists(NORMALIZATION_PATH) and os.path.exists(CLASSES_PATH)):
        print("Error: PyTorch CNN model files not found! Please run train_cnn.py first.")
        return

    # Check input file
    if not os.path.exists(args.file):
        print(f"Error: Input file '{args.file}' does not exist.")
        return

    print("Loading model components...")
    classes = joblib.load(CLASSES_PATH)
    norm_params = joblib.load(NORMALIZATION_PATH)
    mean = norm_params['mean']
    std = norm_params['std']

    # Initialize model and load weights
    model = EmotionCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    print(f"Extracting 2D features from {args.file}...")
    features = extract_mel_spectrogram(args.file)
    if features is None:
        print("Failed to process the audio file.")
        return

    # Normalize using training mean/std
    features = (features - mean) / (std + 1e-7)

    # Convert to PyTorch tensor with dimensions: (Batch=1, Channel=1, Height=128, Width=128)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        prediction_idx = np.argmax(probabilities)
        prediction = classes[prediction_idx]

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
