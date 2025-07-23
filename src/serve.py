# src/serve.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
from google.cloud import storage
import shutil
import pickle

app = Flask(__name__)

# 1) Determine where Vertex AI has staged your model artifacts
gcs_model_uri = os.environ.get("AIP_STORAGE_URI")
local_base_dir = "/model"
client = None
bucket = None
prefix = None
bucket_name = None

# Download model + scaler from GCS if running in Vertex AI
if gcs_model_uri:
    client = storage.Client()
    bucket_name, prefix = gcs_model_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    # Clean out local directory
    if os.path.isdir(local_base_dir):
        shutil.rmtree(local_base_dir)
    os.makedirs(local_base_dir, exist_ok=True)

    # Download all files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(local_base_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
else:
    # In non-GCP context, assume model+scaler are baked into /model
    print("AIP_STORAGE_URI not set; expecting model under /model")

# 2) Define paths
model_dir = os.path.join(local_base_dir, "tf_model")
scaler_path = os.path.join(local_base_dir, "scaler.pkl")

# 3) Load the Keras model
model = tf.keras.models.load_model(model_dir)

# 4) Load the scaler
if not os.path.isfile(scaler_path):
    raise FileNotFoundError(f"scaler.pkl not found at {scaler_path}")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# 5) Class map
CLASS_MAP = {0: 'h', 1: 'l', 2: 'n'}

# 6) Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json(force=True)
    instances = req.get('instances')
    if instances is None:
        return jsonify({"error": "Request JSON must have an 'instances' key."}), 400

    # Ensure a 2D array
    data = np.array(instances, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    # Preprocess
    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)

    # Decode
    if preds.ndim == 2 and preds.shape[1] > 1:
        idxs = np.argmax(preds, axis=1)
        labels = [CLASS_MAP[i] for i in idxs]
    else:
        labels = preds.flatten().tolist()

    return jsonify({"predictions": labels})

# Health check
@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
