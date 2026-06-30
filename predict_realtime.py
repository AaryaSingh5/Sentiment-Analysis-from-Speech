import os
import time
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
import sounddevice as sd
from scipy.io import wavfile

# Paths to models
MODEL_PATH = "emotion_crnn_model.pth"
NORMALIZATION_PATH = "normalization_params_crnn.joblib"
CLASSES_PATH = "classes_crnn.joblib"
TEMP_WAV = "live_recording.wav"

# Define Self-Attention Module
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_outputs):
        scores = self.attention_weights(lstm_outputs)
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        return context_vector, attention_weights

# Define CRNN + Attention Architecture
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

def record_audio(duration=3.0, sample_rate=22050):
    """
    Records audio from the microphone and saves it as a WAV file.
    """
    print("\n* Recording starting in...")
    for countdown in [3, 2, 1]:
        print(f"  {countdown}...")
        time.sleep(1)
        
    print("🔴 RECORDING... Speak now!")
    # Record float32 mono channel audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is done
    print("⏹️ Recording finished!")
    
    # Save as WAV file
    wavfile.write(TEMP_WAV, sample_rate, recording)
    print(f"Saved recording to {TEMP_WAV}")

def run_inference(model, classes, mean, std):
    """
    Loads the saved recording and predicts emotion using CRNN + Attention.
    """
    if not os.path.exists(TEMP_WAV):
        print("Error: No recording file found.")
        return

    features = extract_mel_spectrogram(TEMP_WAV)
    if features is None:
        print("Failed to process the recorded audio.")
        return

    # Normalize
    features = (features - mean) / (std + 1e-7)

    # Convert to PyTorch tensor (Batch=1, Channel=1, 128, 128)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs, attn_weights = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        prediction_idx = np.argmax(probabilities)
        prediction = classes[prediction_idx]
        attention = attn_weights.squeeze().numpy()

    # Display results
    print("\n" + "="*50)
    print(f"🔮 PREDICTED EMOTION: {prediction.upper()}")
    print("="*50)
    print("Confidence Scores:")
    for class_name, prob in zip(classes, probabilities):
        print(f" - {class_name.capitalize():<18}: {prob * 100:.2f}%")
    print("="*50)
    
    # Print the attention timeline
    print("Attention Timeline (Where the model focused its attention):")
    time_per_step = 4 * 512 / 22050
    max_attention = attention.max() if attention.max() > 0 else 1.0
    for i, w in enumerate(attention):
        timestamp = i * time_per_step
        bar_len = int((w / max_attention) * 20)
        bar = '#' * bar_len
        print(f" {timestamp:4.2f}s: {bar:<20} ({w * 100:.1f}%)")
    print("="*50 + "\n")

def main():
    # Verify model files
    if not (os.path.exists(MODEL_PATH) and os.path.exists(NORMALIZATION_PATH) and os.path.exists(CLASSES_PATH)):
        print("Error: Model components not found! Please train train_crnn.py first.")
        return

    print("Loading CRNN + Attention model components...")
    classes = joblib.load(CLASSES_PATH)
    norm_params = joblib.load(NORMALIZATION_PATH)
    mean = norm_params['mean']
    std = norm_params['std']

    # Initialize model
    model = EmotionCRNN(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    print("\n" + "★"*50)
    print("  Real-time Speech Emotion Detection Client")
    print("★"*50)
    print("This tool will record 3 seconds of your voice and predict your emotion.")
    
    try:
        while True:
            cmd = input("Press ENTER to start recording (or type 'q' to exit): ").strip()
            if cmd.lower() == 'q':
                print("Exiting real-time client. Goodbye!")
                break
            
            # Record and predict
            record_audio(duration=3.0)
            run_inference(model, classes, mean, std)
            
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")

if __name__ == "__main__":
    main()
