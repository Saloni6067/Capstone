import os
import argparse
import numpy as np
import pandas as pd
from io import BytesIO
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sk_split
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def upload_to_gcs(source_path: str, bucket_name: str, destination_path: str, recursive=False):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if recursive and os.path.isdir(source_path):
        for root, _, files in os.walk(source_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, source_path)
                blob = bucket.blob(os.path.join(destination_path, relative_path))
                blob.upload_from_filename(full_path)
    else:
        blob = bucket.blob(destination_path)
        blob.upload_from_filename(source_path)

def train_and_evaluate(df: pd.DataFrame, model_dir: str, bucket: str=None,
                       epochs: int=10, batch_size: int=256):
    df = df.drop(columns=['satellite', 'instrument', 'version', 'acq_date', 'acq_time'], errors='ignore')
    df['daynight'] = df['daynight'].map({'D': 0, 'N': 1})
    
    target_col = 'frp'
    feature_cols = ['latitude', 'longitude', 'brightness', 'bright_t31', 'scan', 'track', 'daynight', 'type']
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train_full, X_test, y_train_full, y_test = sk_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = sk_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, dtype='float32')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], jit_compile=True)
    
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early], verbose=1)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    y_pred = model.predict(test_ds).flatten()
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    saved_model_path = os.path.join(model_dir, 'tf_model')
    model.save(saved_model_path)
    
    scaler_path = os.path.join(saved_model_path, 'scaler.pkl')
    pd.to_pickle(scaler, scaler_path)

    if bucket:
        upload_to_gcs(saved_model_path, bucket, 'model/tf_model', recursive=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, help='GCS bucket name')
    parser.add_argument('--data_path', type=str, default='data/NASA.csv', help='GCS path to CSV')
    parser.add_argument('--model_dir', type=str, default='model', help='Local model directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    if args.bucket:
        local_path = '/tmp/NASA.csv'
        download_from_gcs(args.bucket, args.data_path, local_path)
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(args.data_path)

    train_and_evaluate(df, model_dir=args.model_dir, bucket=args.bucket,
                       epochs=args.epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)
    main()
