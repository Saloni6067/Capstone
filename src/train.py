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
    """
    Download all objects under `prefix` in S3 `bucket` into local_dir.
    Expects your CSV at s3://bucket/prefix/NASA.csv
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=prefix):
        # skip "folders"
        if obj.key.endswith('/'):
            continue
        target_path = os.path.join(local_dir, os.path.relpath(obj.key, prefix))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        bucket.download_file(obj.key, target_path)
    return local_dir

def load_data(data_dir: str, filename: str = "NASA.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df

def train_and_evaluate(df: pd.DataFrame, model_dir: str):
    # --- 1. Drop unused columns ---
    df = df.drop(columns=['satellite', 'instrument', 'version'])

    # --- 2. Encode categorical columns ---
    le = LabelEncoder()
    df['confidence_N'] = le.fit_transform(df['confidence'])
    df = pd.get_dummies(df, columns=['daynight'], drop_first=True)

    # --- 3. Select features & target ---
    features = [
        'latitude','longitude','brightness','bright_t31','frp',
        'scan','track','daynight_N','type'
    ]
    X = df[features]
    y = df['confidence_N']

    # --- 4. Balance classes with SMOTE ---
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # --- 5. Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # --- 6. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    # --- 7. Build model ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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

    # --- 8. Train with early stopping ---
    early_stop = EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # --- 9. Evaluate & report ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # --- 10. Save artifacts ---
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'tf_model'))
    pd.to_pickle(scaler, os.path.join(model_dir, 'scaler.pkl'))
    # Plot loss curve
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'loss_curve.png'))
    plt.close()
    print(f"\nSaved model + scaler + plot to {model_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Train NASA.csv model from S3 and save artifacts"
    )
    parser.add_argument(
        "--bucket", required=True,
        help="S3 bucket name containing your data"
    )
    parser.add_argument(
        "--prefix", default="data/",
        help="S3 prefix under the bucket where NASA.csv lives"
    )
    parser.add_argument(
        "--model-dir", default="/opt/ml/model",
        help="Directory to save model artifacts (TF model, scaler, plots)"
    )
    args = parser.parse_args()

    # Download to a temp folder
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading s3://{args.bucket}/{args.prefix} → {tmp}")
        download_from_s3(args.bucket, args.prefix, tmp)
        df = load_data(tmp, filename="NASA.csv")

    # Train & save
    train_and_evaluate(df, args.model_dir)

if __name__ == "__main__":
    main()
