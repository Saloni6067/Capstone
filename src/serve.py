# src/serve.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
from google.cloud import storage
import shutil
import joblib  # for loading scaler

app = Flask(__name__)

# Vertex sets AIP_STORAGE_URI to gs://…/model/tf_model
gcs_model_uri = os.environ.get("AIP_STORAGE_URI")
local_model_dir = "/model"        # target root
local_tf_dir = os.path.join(local_model_dir, "tf_model")
local_scaler_path = os.path.join(local_tf_dir, "scaler.pkl")

if gcs_model_uri:
    client = storage.Client()
    bucket_name, prefix = gcs_model_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    # clean and download everything under tf_model prefix into /model/tf_model
    if os.path.isdir(local_model_dir):
        shutil.rmtree(local_model_dir)
    os.makedirs(local_tf_dir, exist_ok=True)

    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = os.path.relpath(blob.name, prefix)
        dest = os.path.join(local_tf_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
else:
    print(" AIP_STORAGE_URI not set; expecting /model/tf_model baked into image")

# load model + scaler
model = tf.keras.models.load_model(local_tf_dir)
if not os.path.isfile(local_scaler_path):
    raise FileNotFoundError(f"scaler.pkl not found at {local_scaler_path}")
scaler = joblib.load(local_scaler_path)

CLASS_MAP = {0: 'h', 1: 'l', 2: 'n'}

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json(force=True)
    instances = req.get("instances")
    if instances is None:
        return jsonify({"error": "Missing 'instances'"}), 400

    x = np.array(instances, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]         # ensure batch dim

    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    idxs = np.argmax(preds, axis=1)
    labels = [CLASS_MAP[i] for i in idxs]

    return jsonify({"predictions": labels})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
