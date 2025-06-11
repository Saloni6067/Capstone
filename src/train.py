#!/usr/bin/env python
# src/train.py

import argparse
import os
import tempfile

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

def download_from_s3(bucket: str, prefix: str, local_dir: str):
    """Download all objects under prefix in bucket to local_dir."""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith('/'):
            continue
        target = os.path.join(local_dir, os.path.relpath(obj.key, prefix))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        bucket.download_file(obj.key, target)
    return local_dir

def load_data(local_dir: str, filename: str = "NASA.csv") -> pd.DataFrame:
    path = os.path.join(local_dir, filename)
    return pd.read_csv(path)

def build_and_train(df: pd.DataFrame, output_model_dir: str):
    # --- Preprocessing ---
    # Drop unused columns
    df = df.drop(columns=['satellite', 'version', 'instrument'])
    
    # Encode confidence
    le = LabelEncoder()
    df['confidence_N'] = le.fit_transform(df['confidence'])
    
    # One-hot encode daynight
    df = pd.get_dummies(df, columns=['daynight'], drop_first=True)
    
    # Feature / target split
    features = ['brightness', 'bright_t31', 'frp_capped', 'scan',
                'track', 'daynight_N', 'type', 'latitude', 'longitude']
    X = df[features]
    y = df['confidence_N']
    
    # Handle imbalance
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )
    
    # --- Model ---
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # --- Evaluation ---
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # --- Save model & scaler ---
    os.makedirs(output_model_dir, exist_ok=True)
    model.save(os.path.join(output_model_dir, "tf_model"))
    # Save the scaler so you can apply same preprocessing at inference
    pd.to_pickle(scaler, os.path.join(output_model_dir, "scaler.pkl"))
    print(f"\nModel and scaler saved to {output_model_dir}")
    
    # --- Plot loss ---
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model_dir, "loss_plot.png"))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow model on NASA.csv from S3")
    parser.add_argument("--bucket", required=True,
                        help="S3 bucket name where the data lives")
    parser.add_argument("--prefix", default="data/",
                        help="S3 prefix under the bucket for NASA.csv")
    parser.add_argument("--model-dir", default="/opt/ml/model",
                        help="Local directory to save the trained model and artifacts")
    args = parser.parse_args()
    
    # 1. Fetch data
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading s3://{args.bucket}/{args.prefix} → {tmpdir}")
        download_from_s3(args.bucket, args.prefix, tmpdir)
        df = load_data(tmpdir, filename="NASA.csv")
    
    # 2. Train & save
    build_and_train(df, args.model_dir)

if __name__ == "__main__":
    main()
