#!/usr/bin/env python
import os
import pickle
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.cloud import storage

app = Flask(__name__)

# Where to download / serve the model inside the container
MODEL_PATH = "/model/tf_model"

# Environment variables set in your Dockerfile
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "RegModel")

def fetch_from_gcs(bucket_name: str, prefix: str, dest_dir: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # List all blobs under the tf_model/ folder
    for blob in bucket.list_blobs(prefix=f"{prefix}/tf_model/"):
        if blob.name.endswith("/"):
            continue
        filename = os.path.basename(blob.name)
        local_path = os.path.join(dest_dir, filename)
        os.makedirs(dest_dir, exist_ok=True)
        blob.download_to_filename(local_path)

# On startup, download the SavedModel folder if not already present
if GCS_BUCKET:
    if not os.path.exists(os.path.join(MODEL_PATH, "saved_model.pb")):
        fetch_from_gcs(GCS_BUCKET, GCS_PREFIX, MODEL_PATH)

# Now load the model directory (not the .pb file directly)
model = keras.models.load_model(MODEL_PATH)
# Load the scaler or preprocessor
with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    instances = np.array(data['instances'])
    proc = preprocessor.transform(instances)
    preds = model.predict(proc).flatten().tolist()
    return jsonify({'predictions': preds})

if __name__ == '__main__':
    # Use default port or override with PORT env var
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
