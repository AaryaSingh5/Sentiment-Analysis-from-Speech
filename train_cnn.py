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
MODEL_PATH = "emotion_cnn_model.pth"
NORMALIZATION_PATH = "normalization_params.joblib"

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        # Add channel dimension: (N, 1, 128, 128)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define 2D CNN Architecture
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        
        # Conv block 1: Input (1, 128, 128) -> Output (16, 64, 64)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv block 2: Input (16, 64, 64) -> Output (32, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv block 3: Input (32, 32, 32) -> Output (64, 16, 16)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Conv block 4: Input (64, 16, 16) -> Output (128, 8, 8)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully Connected layers
        # Flattened features: 128 channels * 8 height * 8 width = 8192
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
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def train():
    if not os.path.exists(INPUT_NPZ):
        print(f"Error: {INPUT_NPZ} not found. Please run preprocess_cnn.py first.")
        return

    # Load preprocessed features
    print("Loading preprocessed dataset...")
    data = np.load(INPUT_NPZ)
    X = data['X']
    y = data['y']
    classes = data['classes']
    num_classes = len(classes)
    print(f"Dataset shape: {X.shape}, classes: {classes}")

    # Normalize globally (mean=0, std=1)
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-7)
    
    # Save normalization parameters
    print(f"Saving normalization parameters to {NORMALIZATION_PATH}...")
    joblib.dump({'mean': mean, 'std': std}, NORMALIZATION_PATH)

    # Save classes
    print("Saving classes list to classes_cnn.joblib...")
    joblib.dump(classes, "classes_cnn.joblib")

    # Train/Validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Create PyTorch datasets & dataloaders
    train_dataset = SpectrogramDataset(X_train, y_train)
    val_dataset = SpectrogramDataset(X_val, y_val)
    
    # Batch size is set to 64 for moderate memory usage
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss, optimizer
    model = EmotionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training configuration
    epochs = 15
    best_val_loss = float('inf')
    early_stop_patience = 4
    no_improvement_epochs = 0

    print("Beginning model training...")
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
            # Save best weights
            torch.save(model.state_dict(), MODEL_PATH)
            print(f" ==> Saved new best model checkpoint to {MODEL_PATH}")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stop_patience:
                print(f"Early stopping triggered! No improvement in val loss for {early_stop_patience} epochs.")
                break

    print("Training process finished!")

if __name__ == "__main__":
    train()
