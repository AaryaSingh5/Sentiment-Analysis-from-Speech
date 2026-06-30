import os
import argparse
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn

# Paths to models
MODEL_PATH = "emotion_crnn_model.pth"
NORMALIZATION_PATH = "normalization_params_crnn.joblib"
CLASSES_PATH = "classes_crnn.joblib"

# Define Self-Attention Module (Must match training exactly)
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_outputs):
        scores = self.attention_weights(lstm_outputs)
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        return context_vector, attention_weights

# Define CRNN + Attention Architecture (Must match training exactly)
class EmotionCRNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.lstm = nn.LSTM(
            input_size=128 * 8, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.attention = Attention(hidden_dim=256)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)
        x = self.dropout(torch.relu(self.fc1(context_vector)))
        x = self.fc2(x)
        return x, attn_weights

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
    parser = argparse.ArgumentParser(description="Predict emotion from a speech WAV file using Hybrid CRNN + Attention.")
    parser.add_argument("--file", type=str, required=True, help="Path to the audio WAV file.")
    args = parser.parse_args()

    # Check model files
    if not (os.path.exists(MODEL_PATH) and os.path.exists(NORMALIZATION_PATH) and os.path.exists(CLASSES_PATH)):
        print("Error: CRNN model files not found! Please run train_crnn.py first.")
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
    model = EmotionCRNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    print(f"Extracting features from {args.file}...")
    features = extract_mel_spectrogram(args.file)
    if features is None:
        print("Failed to process the audio file.")
        return

    # Normalize
    features = (features - mean) / (std + 1e-7)

    # Convert to PyTorch tensor with dimensions: (Batch=1, Channel=1, Height=128, Width=128)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs, attn_weights = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        prediction_idx = np.argmax(probabilities)
        prediction = classes[prediction_idx]
        
        # Squeeze attention weights: shape becomes (32,)
        attention = attn_weights.squeeze().numpy()

    # Display results
    print("\n" + "="*50)
    print(f"Predicted Emotion: {prediction.upper()}")
    print("="*50)
    print("Confidence Scores:")
    for class_name, prob in zip(classes, probabilities):
        print(f" - {class_name.capitalize():<18}: {prob * 100:.2f}%")
    print("="*50)
    
    # Print the attention timeline
    print("Attention Timeline (Where the model focused its attention):")
    # TESS and CREMA-D audios are resampled to 22050. The width is 128 frames.
    # MaxPool reduces the width to 32 steps.
    # Each time step represents 128 / 32 = 4 spectrogram frames.
    # 4 frames * 512 hop_length / 22050 sr = 0.093 seconds per step.
    time_per_step = 4 * 512 / 22050
    
    # Scale attention scores for printing (max attention gets 20 blocks)
    max_attention = attention.max() if attention.max() > 0 else 1.0
    for i, w in enumerate(attention):
        timestamp = i * time_per_step
        bar_len = int((w / max_attention) * 20)
        bar = '#' * bar_len
        print(f" {timestamp:4.2f}s: {bar:<20} ({w * 100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
