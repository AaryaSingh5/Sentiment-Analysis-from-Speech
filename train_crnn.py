import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Paths
INPUT_NPZ = "extracted_features_cnn.npz"
MODEL_PATH = "emotion_crnn_model.pth"
NORMALIZATION_PATH = "normalization_params_crnn.joblib"
CLASSES_PATH = "classes_crnn.joblib"

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # shape: (N, 1, 128, 128)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Self-Attention Module
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_outputs):
        # lstm_outputs shape: (Batch, SeqLen, HiddenDim)
        # Calculate attention scores for each time frame
        scores = self.attention_weights(lstm_outputs) # Shape: (Batch, SeqLen, 1)
        # Softmax over sequence/time dimension (dim=1) to get weights summing to 1
        attention_weights = torch.softmax(scores, dim=1) # Shape: (Batch, SeqLen, 1)
        # Context vector is the weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1) # Shape: (Batch, HiddenDim)
        return context_vector, attention_weights

# Define CRNN + Attention Architecture
class EmotionCRNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCRNN, self).__init__()
        
        # 1. CNN Feature Extractor
        # Conv block 1: Input (1, 128, 128) -> Output (16, 64, 128)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1)) # pools frequency, keeps time
        
        # Conv block 2: Input (16, 64, 128) -> Output (32, 32, 128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1)) # pools frequency, keeps time
        
        # Conv block 3: Input (32, 32, 128) -> Output (64, 16, 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2)) # pools both frequency and time
        
        # Conv block 4: Input (64, 16, 64) -> Output (128, 8, 32)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2)) # pools both frequency and time
        
        # 2. Recurrent Layer (Bidirectional LSTM)
        # Sequence input: 32 time frames, each having 128 channels * 8 height = 1024 features
        self.lstm = nn.LSTM(
            input_size=128 * 8, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Self-Attention Layer
        # hidden_size * 2 because BiLSTM outputs 128 forward + 128 backward = 256
        self.attention = Attention(hidden_dim=256)
        
        # 4. Dense Classifier
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x)))) # Shape: (Batch, 128, 8, 32)
        
        # Re-arrange dimensions for recurrent input: (Batch, SeqLen=32, InputSize=1024)
        # Permute (Batch, 128, 8, 32) -> (Batch, 32, 128, 8)
        x = x.permute(0, 3, 1, 2)
        # Flatten height and channels: (Batch, 32, 128 * 8)
        x = x.reshape(x.size(0), x.size(1), -1)
        
        # Bidirectional LSTM sequence modeling
        lstm_out, _ = self.lstm(x) # Shape: (Batch, 32, 256)
        
        # Self-Attention pooling over time frames
        context_vector, attn_weights = self.attention(lstm_out) # Shape: (Batch, 256)
        
        # Classification
        x = self.dropout(torch.relu(self.fc1(context_vector)))
        x = self.fc2(x)
        return x

def train():
    if not os.path.exists(INPUT_NPZ):
        print(f"Error: {INPUT_NPZ} not found. Please run preprocess_cnn.py first.")
        return

    # Load dataset
    print("Loading preprocessed dataset...")
    data = np.load(INPUT_NPZ)
    X = data['X']
    y = data['y']
    classes = data['classes']
    num_classes = len(classes)
    print(f"Dataset loaded: {X.shape}, classes: {classes}")

    # Standardize spectrogram features
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-7)
    
    # Save normalization parameters
    print(f"Saving normalization parameters to {NORMALIZATION_PATH}...")
    joblib.dump({'mean': mean, 'std': std}, NORMALIZATION_PATH)

    # Save classes list
    print(f"Saving classes list to {CLASSES_PATH}...")
    joblib.dump(classes, CLASSES_PATH)

    # Train/Validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Create PyTorch datasets & dataloaders
    train_dataset = SpectrogramDataset(X_train, y_train)
    val_dataset = SpectrogramDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, optimizer, loss
    model = EmotionCRNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    best_val_loss = float('inf')
    early_stop_patience = 4
    no_improvement_epochs = 0

    print("Beginning CRNN + Attention model training...")
    for epoch in range(1, epochs + 1):
        # 1. Train step
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # 2. Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

        # 3. Model checkpointing & early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f" ==> Saved new best model checkpoint to {MODEL_PATH}")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stop_patience:
                print(f"Early stopping triggered! No improvement in val loss for {early_stop_patience} epochs.")
                break

    print("CRNN Training pipeline finished successfully!")

if __name__ == "__main__":
    train()
