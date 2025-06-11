# src/train.py
#!/usr/bin/env python

import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        file_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(file_path)
    return local_dir


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)


def load_data(data_dir: str, filename: str = "NASA.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df


def train_and_evaluate(df: pd.DataFrame, model_dir: str, bucket: str = None):
    # Drop unused columns
    df = df.drop(columns=['satellite', 'instrument', 'version'])
    # Encode target
    le = LabelEncoder()
    df['confidence_N'] = le.fit_transform(df['confidence'])
    df = pd.get_dummies(df, columns=['daynight'], drop_first=True)

    # Features and target
    features = ['latitude', 'longitude', 'brightness', 'bright_t31', 'frp',
                'scan', 'track', 'daynight_N', 'type']
    X = df[features]
    y = df['confidence_N']

    # Balance data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(np.unique(y_res)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    saved_model_dir = os.path.join(model_dir, 'tf_model')
    model.save(saved_model_dir)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    pd.to_pickle(scaler, scaler_path)

    # Plot loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(model_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Upload to GCS
    if bucket:
        print(f"Uploading artifacts to {bucket}/model/")
        upload_to_gcs(os.path.join(saved_model_dir, 'saved_model.pb'), bucket, 'model/saved_model.pb')
        upload_to_gcs(scaler_path, bucket, 'model/scaler.pkl')
        upload_to_gcs(loss_plot_path, bucket, 'model/loss_curve.png')


def main():
    parser = argparse.ArgumentParser(description="Train NASA model from GCS and upload artifacts")
    parser.add_argument("--bucket", default="capstone-nasa-wildfire-sal", help="GCS bucket for data and models")
    parser.add_argument("--prefix", default="data/", help="GCS prefix for input data")
    parser.add_argument("--model-dir", default="temp/model", help="Local dir to save artifacts")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading from gs://{args.bucket}/{args.prefix} to {tmp}")
        download_from_gcs(args.bucket, args.prefix, tmp)
        df = load_data(tmp)

    train_and_evaluate(df, args.model_dir, args.bucket)

if __name__ == "__main__":
    main()