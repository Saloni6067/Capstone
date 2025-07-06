#!/usr/bin/env python

import argparse
import os
import tempfile
import tensorflow as tf
from tensorflow import keras
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


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str, recursive: bool = False):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if recursive and os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, local_path)
                blob = bucket.blob(os.path.join(gcs_path, rel_path).replace('\\', '/'))
                blob.upload_from_filename(full_path)
                print(f"Uploaded {full_path} to gs://{bucket_name}/{gcs_path}/{rel_path}")
    else:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")


def load_data(data_dir: str, filename: str = "NASA.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    print(f"Loading data from {path}")
    return pd.read_csv(path)


def train_and_evaluate(df: pd.DataFrame, model_dir: str, bucket: str = None):
    # Preprocessing
    df = df.drop(columns=['satellite', 'instrument', 'version'])
    le = LabelEncoder()
    df['confidence_N'] = le.fit_transform(df['confidence'])
    df = pd.get_dummies(df, columns=['daynight'], drop_first=True)

    features = ['latitude', 'longitude', 'brightness', 'bright_t31', 'frp',
                'scan', 'track', 'daynight_N', 'type']
    X, y = df[features], df['confidence_N']

    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_scaled = StandardScaler().fit_transform(X_res)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    # Model building
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(np.unique(y_res)), activation='softmax')
    ])
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=16,
        callbacks=[EarlyStopping('val_loss', patience=6, restore_best_weights=True)],
        verbose=1
    )

    print("\nClassification Report:")
    print(classification_report(y_test, np.argmax(model.predict(X_test), axis=1),
                                target_names=le.classes_))

    # Save locally
    os.makedirs(model_dir, exist_ok=True)
    saved_model_dir = os.path.join(model_dir, 'tf_model')
    model.save(saved_model_dir)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    pd.to_pickle(StandardScaler().fit(X_res), scaler_path)

    # Loss plot
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    loss_plot = os.path.join(model_dir, 'loss_curve.png')
    plt.savefig(loss_plot); plt.close()

    # Upload to GCS
    if bucket:
        print(f"Uploading to gs://{bucket}/model/tf_model/")
        upload_to_gcs(saved_model_dir, bucket, 'model/tf_model', recursive=True)
        upload_to_gcs(scaler_path, bucket, 'model/scaler.pkl')
        upload_to_gcs(loss_plot, bucket, 'model/loss_curve.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket', default='capstone-nasa-wildfire-sal')
    p.add_argument('--prefix', default='data/')
    p.add_argument('--model-dir', default='temp/model')
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        download_from_gcs(args.bucket, args.prefix, tmp)
        df = load_data(tmp)
    train_and_evaluate(df, args.model_dir, args.bucket)


if __name__ == '__main__':
    main()