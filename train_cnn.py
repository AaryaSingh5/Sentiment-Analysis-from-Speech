import os
import random
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

# Define Dataset with SpecAugment data augmentation
class SpectrogramDataset(Dataset):
    def __init__(self, X, y, augment=False):
        # Add channel dimension: (N, 1, 128, 128)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = self._spec_augment(x)
        return x, y

    def _spec_augment(self, spec, num_mask=2, freq_masking_max=15, time_masking_max=20):
        # spec shape: (1, 128, 128)
        c, h, w = spec.shape
        augmented = spec.clone()
        
        # Mask with the minimum value (silence representation in log-mel spectrogram)
        mask_val = spec.min().item()
        
        for _ in range(num_mask):
            # Frequency masking
            f = random.randint(0, freq_masking_max)
            f0 = random.randint(0, h - f)
            augmented[:, f0:f0+f, :] = mask_val
            
            # Time masking
            t = random.randint(0, time_masking_max)
            t0 = random.randint(0, w - t)
            augmented[:, :, t0:t0+t] = mask_val
            
        return augmented

# Define Residual Block for CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define Lightweight 2D CNN with Residual Connections
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        
        # Initial Conv block: Input (1, 128, 128) -> Output (32, 64, 64)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Residual Blocks
        self.res1 = ResidualBlock(32, 32)
        self.pool2 = nn.MaxPool2d(2, 2) # -> (32, 32, 32)
        
        self.res2 = ResidualBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2) # -> (64, 16, 16)
        
        self.res3 = ResidualBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2) # -> (128, 8, 8)
        
        self.res4 = ResidualBlock(128, 256)
        
        # Global Average Pooling: (256, 8, 8) -> (256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.res1(x))
        x = self.pool3(self.res2(x))
        x = self.pool4(self.res3(x))
        x = self.res4(x)
        
        # Global Average Pooling & Flatten
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
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
    train_dataset = SpectrogramDataset(X_train, y_train, augment=True)
    val_dataset = SpectrogramDataset(X_val, y_val, augment=False)
    
    # Batch size of 64
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss, optimizer, scheduler
    model = EmotionCNN(num_classes=num_classes).to(device)
    
    # Label Smoothing regularizes the classifications
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    early_stop_patience = 5
    no_improvement_epochs = 0

    print("Beginning upgraded ResNet model training...")
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

        # Step Learning Rate Scheduler
        scheduler.step()

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

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | "
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
