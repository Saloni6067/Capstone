# train_reg.py
#!/usr/bin/env python
import argparse
import os
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from google.cloud import storage
import pickle


def download_from_gcs(bucket_name: str, prefix: str, local_dir: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith('/'):
            continue
        dest = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(dest)
    return local_dir


def load_data(bucket: str, prefix: str, filename: str='NASA.csv') -> pd.DataFrame:
    if bucket:
        with tempfile.TemporaryDirectory() as tmp:
            download_from_gcs(bucket, prefix, tmp)
            return pd.read_csv(os.path.join(tmp, filename))
    else:
        return pd.read_csv(os.path.join(prefix, filename))


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default=None)
    parser.add_argument('--prefix', default='RegModel')
    parser.add_argument('--model-dir', default='/tmp/model')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    df = load_data(args.bucket, args.prefix, 'NASA.csv')
    X = df.drop(columns=['frp'])
    y = df['frp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'bright_t31']
    categorical_features = ['satellite', 'instrument', 'daynight', 'type']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model = build_model(X_train_proc.shape[1])
    early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(
        X_train_proc, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop], verbose=1
    )

    y_pred = model.predict(X_test_proc).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    saved_path = os.path.join(args.model_dir, 'tf_model')
    model.save(saved_path)
    with open(os.path.join(saved_path, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)

    if args.bucket:
        client = storage.Client()
        bucket = client.bucket(args.bucket)
        for fname in os.listdir(saved_path):
            local_path = os.path.join(saved_path, fname)
            blob = bucket.blob(f"{args.prefix}/tf_model/{fname}")
            blob.upload_from_filename(local_path)

if __name__ == '__main__':
    main()