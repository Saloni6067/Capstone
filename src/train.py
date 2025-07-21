#!/usr/bin/env python

import argparse
import os
import tempfile
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# Enable mixed precision and XLA for performance
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

def stratified_sample_with_min(df, label_col, frac=0.02, min_per_class=50, random_state=42):
    """
    Take a stratified sample ensuring at least min_per_class examples per label.
    """
    pieces = []
    for cls, grp in df.groupby(label_col):
        n = min(max(int(len(grp) * frac), min_per_class), len(grp))
        pieces.append(grp.sample(n=n, random_state=random_state))
    return pd.concat(pieces).reset_index(drop=True)


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith('/'):
            continue
        dest = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(dest)
    return local_dir


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str, recursive: bool=False):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if recursive and os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, local_path)
                blob = bucket.blob(os.path.join(gcs_path, rel).replace('\\','/'))
                blob.upload_from_filename(src)
    else:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)


def load_data_gcs(bucket: str, prefix: str, filename: str = 'NASA.csv') -> pd.DataFrame:
    # Download prefix to temp and load
    with tempfile.TemporaryDirectory() as tmp:
        download_from_gcs(bucket, prefix, tmp)
        path = os.path.join(tmp, filename)
        print(f"Loading data from gs://{bucket}/{prefix}{filename}")
        return pd.read_csv(path)


def load_data_local(data_dir: str, filename: str = 'NASA.csv') -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    print(f"Loading data from {path}")
    return pd.read_csv(path)


def train_and_evaluate(df: pd.DataFrame, model_dir: str, bucket: str=None,
                       epochs: int=10, batch_size: int=256):
    # 1) Stratified sample
    df = stratified_sample_with_min(df, label_col='confidence', frac=0.02, min_per_class=100)

    # 2) Drop irrelevant columns
    df = df.drop(columns=['satellite','instrument','version','acq_date','acq_time'], errors='ignore')

    # 3) Cap outliers
    df['frp'] = df['frp'].clip(upper=2000)

    # 4) Encode labels
    le = LabelEncoder()
    df['confidence_code'] = le.fit_transform(df['confidence'])

    # 5) Map day/night
    df['daynight'] = df['daynight'].map({'D':0,'N':1})

    # 6) Features & target
    feature_cols = ['latitude','longitude','brightness','bright_t31','frp',
                    'scan','track','daynight','type']
    X, y = df[feature_cols].values, df['confidence_code'].values

    # 7) SMOTE
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    unique, counts = np.unique(y_res, return_counts=True)
    print("Class distribution after SMOTE:")
    for cls, cnt in zip(unique, counts):
        print(f"  {le.inverse_transform([cls])[0]}: {cnt}")

    # 8) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # 9) Split train/val/test
    X_train_full, X_test, y_train_full, y_test = sk_split(
        X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    X_train, X_val, y_train, y_val = sk_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    # 10) tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 11) Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(le.classes_), activation='softmax', dtype='float32')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )

    # 12) Train
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=[early], verbose=1)

    # 13) Evaluate
    print("\nClassification Report on Test Set:")
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 14) Save
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'tf_model'))
    pd.to_pickle(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # 15) Plot
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(model_dir, 'loss_curve.png'))
    plt.close()

    # 16) Upload if bucket provided
    if bucket:
        upload_to_gcs(os.path.join(model_dir, 'tf_model'), bucket, 'model/tf_model', recursive=True)
        upload_to_gcs(os.path.join(model_dir, 'scaler.pkl'), bucket, 'model/scaler.pkl')
        upload_to_gcs(os.path.join(model_dir, 'loss_curve.png'), bucket, 'model/loss_curve.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default=None, help='GCS bucket name')
    parser.add_argument('--prefix', default='', help='GCS prefix')
    parser.add_argument('--model-dir', default='temp/model', help='Output model directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    # Load data from GCS if bucket provided, otherwise local data directory
    if args.bucket:
        df = load_data_gcs(args.bucket, args.prefix)
    else:
        df = load_data_local(os.path.join(os.getcwd(), 'data'))

    train_and_evaluate(
        df,
        args.model_dir,
        bucket=args.bucket,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
