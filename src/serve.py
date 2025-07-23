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

if gcs_model_uri:
    # Download model directory and scaler from GCS\    client = storage.Client()
    bucket_name, prefix = gcs_model_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    # Clean local directory
    if os.path.isdir(local_base_dir):
        shutil.rmtree(local_base_dir)
    os.makedirs(local_base_dir, exist_ok=True)

    # Download SavedModel under prefix
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith('/'):
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(local_base_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)

    # Download scaler.pkl from parent folder of prefix
    parent_prefix = os.path.dirname(prefix)
    scaler_blob = bucket.blob(f"{parent_prefix}/scaler.pkl")
    scaler_dest = os.path.join(local_base_dir, 'scaler.pkl')
    scaler_blob.download_to_filename(scaler_dest)
else:
    # Expect /model to contain SavedModel and scaler.pkl baked in
    print("AIP_STORAGE_URI not set; using /model directory as-is")

# 2) Paths for model and scaler
model_dir = local_base_dir
scaler_path = os.path.join(local_base_dir, 'scaler.pkl')

# 3) Load the Keras SavedModel
model = tf.keras.models.load_model(model_dir)

# 4) Load StandardScaler
if not os.path.isfile(scaler_path):
    raise FileNotFoundError(f"scaler.pkl not found at {scaler_path}")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# 5) Class mapping
CLASS_MAP = {0: 'h', 1: 'l', 2: 'n'}

# 6) Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json(force=True)
    instances = req.get('instances')
    if instances is None:
        return jsonify({"error": "Request JSON must have an 'instances' key."}), 400

    data = np.array(instances, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    # Apply the same scaling as in training
    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)

    # Convert softmax to class labels
    idxs = np.argmax(preds, axis=1)
    labels = [CLASS_MAP[i] for i in idxs]

    return jsonify({"predictions": labels})

# 7) Health check
@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
