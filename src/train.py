#!/usr/bin/env python

import argparse
import os
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


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


def upload_file_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")


def upload_dir_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    """Recursively uploads a directory to GCS (used for SavedModel dir)"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            rel_path = os.path.relpath(local_file, local_dir)
            blob_path = os.path.join(gcs_prefix, rel_path).replace("\\", "/")
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_path}")


def load_data(data_dir: str, filename: str = "NASA.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df


def train_and_evaluate(df: pd.DataFrame, model_dir: str, bucket: str = None):
    df = df.drop(columns=['satellite', 'instrument', 'version'])
    le = LabelEncoder()
    df['confidence_N'] = le.fit_transform(df['confidence'])
    df = pd.get_dummies(df, columns=['daynight'], drop_first=True)

    features = ['latitude', 'longitude', 'brightness', 'bright_t31', 'frp',
                'scan', 'track', 'daynight_N', 'type']
    X = df[features]
    y = df['confidence_N']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(np.unique(y_res)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(model_dir, exist_ok=True)
    saved_model_dir = os.path.join(model_dir, 'tf_model')
    model.save(saved_model_dir)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    pd.to_pickle(scaler, scaler_path)

    # Save loss plot
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
        print(f"Uploading artifacts to gs://{bucket}/model/")
        upload_dir_to_gcs(saved_model_dir, bucket, 'model/tf_model')
        upload_file_to_gcs(scaler_path, bucket, 'model/scaler.pkl')
        upload_file_to_gcs(loss_plot_path, bucket, 'model/loss_curve.png')


def main():
    parser = argparse.ArgumentParser(description="Train NASA wildfire model and upload to GCS")
    parser.add_argument("--bucket", default="capstone-nasa-wildfire-sal", help="GCS bucket for data and model")
    parser.add_argument("--prefix", default="data/", help="GCS prefix for input CSV data")
    parser.add_argument("--model-dir", default="temp/model", help="Local directory to store model files")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading from gs://{args.bucket}/{args.prefix} to {tmp}")
        download_from_gcs(args.bucket, args.prefix, tmp)
        df = load_data(tmp)

    train_and_evaluate(df, args.model_dir, args.bucket)


if __name__ == "__main__":
    main()
