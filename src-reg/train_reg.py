# train_reg.py
#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sk_split
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


def stratified_sample(df: pd.DataFrame, target_col: str,
                      n_bins: int = 10,
                      frac: float = 0.02,
                      min_per_bin: int = 50,
                      random_state: int = 42) -> pd.DataFrame:
    df = df.copy()
    df['bin'] = pd.qcut(df[target_col], q=n_bins, duplicates='drop')
    pieces = []
    for _, grp in df.groupby('bin'):
        n = max(int(len(grp) * frac), min_per_bin)
        n = min(n, len(grp))
        pieces.append(grp.sample(n=n, random_state=random_state))
    sampled = pd.concat(pieces).drop(columns=['bin']).reset_index(drop=True)
    return sampled


def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_to_gcs(source_path: str, bucket_name: str, prefix: str, recursive=False):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if recursive and os.path.isdir(source_path):
        for root, _, files in os.walk(source_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel = os.path.relpath(full_path, source_path)
                blob_name = f"{prefix}/tf_model/{rel}"
                bucket.blob(blob_name).upload_from_filename(full_path)
    else:
        blob_name = f"{prefix}/tf_model/{os.path.basename(source_path)}"
        bucket.blob(blob_name).upload_from_filename(source_path)


def train_and_evaluate(df: pd.DataFrame, model_dir: str, bucket: str = None,
                       prefix: str = '', epochs: int = 10, batch_size: int = 256):
    # 1) Stratified sample
    df = stratified_sample(df, 'frp', n_bins=10, frac=0.02, min_per_bin=50)
    # 2) Clean
    df = df.drop(columns=['satellite', 'instrument', 'version', 'acq_date', 'acq_time'], errors='ignore')
    df['daynight'] = df['daynight'].map({'D': 0, 'N': 1})
    # 3) Features & target
    features = ['latitude', 'longitude', 'brightness', 'bright_t31', 'scan', 'track', 'daynight', 'type']
    X, y = df[features].values, df['frp'].values
    # 4) Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 5) Split
    X_train_full, X_test, y_train_full, y_test = sk_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = sk_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
    # 6) tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # 7) Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, dtype='float32')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], jit_compile=True)
    # 8) Train
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early], verbose=1)
    # 9) Evaluate
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    y_pred = model.predict(test_ds).flatten()
    rmse, r2 = (np.sqrt(np.mean((y_test - y_pred) ** 2)), r2_score(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    # 10) Save locally under prefix directory
    os.makedirs(model_dir, exist_ok=True)
    saved_path = os.path.join(model_dir, 'tf_model')
    model.save(saved_path)
    pd.to_pickle(scaler, os.path.join(saved_path, 'scaler.pkl'))
    # 11) Upload only, no overwrite of classification artifacts
    if bucket:
        upload_to_gcs(saved_path, bucket, prefix, recursive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, help='GCS bucket name')
    parser.add_argument('--data_path', type=str, default='data/NASA.csv', help='GCS path to CSV')
    parser.add_argument('--model_dir', type=str, help='Local model directory')
    parser.add_argument('--prefix', type=str, default='RegModel', help='GCS prefix for regression artifacts')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    # derive local model_dir from prefix if not provided
    model_dir = args.model_dir or f"{args.prefix}_model"
    # Load data
    if args.bucket:
        tmp = '/tmp/NASA.csv'
        download_from_gcs(args.bucket, args.data_path, tmp)
        df = pd.read_csv(tmp)
    else:
        df = pd.read_csv(args.data_path)
    train_and_evaluate(df, model_dir=model_dir, bucket=args.bucket,
                       prefix=args.prefix, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(True)
    main()