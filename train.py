import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Paths
INPUT_CSV = "extracted_features.csv"
MODEL_PATH = "emotion_svm_model.joblib"
SCALER_PATH = "scaler.joblib"
CLASSES_PATH = "classes.joblib"

def train_model():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Feature file {INPUT_CSV} not found! Please run preprocess.py first.")
        return

    print(f"Loading features from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Dataset shape: {df.shape}")
    print(f"Distribution of labels:\n{df['label'].value_counts()}")
    print(f"Distribution of datasets:\n{df['dataset'].value_counts()}")

    # Extract features (feat_0, feat_1, ...)
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X = df[feature_cols].values
    y = df['label'].values

    # Stratified train-test split
    print("Splitting dataset into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM classifier
    # Using probability=True so we can output confidence scores during inference
    print("Training Support Vector Machine (SVM) classifier...")
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*50)

    # Save components
    print(f"Saving SVM model to {MODEL_PATH}...")
    joblib.dump(svm, MODEL_PATH)

    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saving unique classes to {CLASSES_PATH}...")
    joblib.dump(svm.classes_, CLASSES_PATH)

    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    train_model()
