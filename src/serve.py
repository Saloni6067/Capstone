# src/serve.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
from google.cloud import storage
import shutil
import joblib

app = Flask(__name__)

# 1) Determine where Vertex AI has staged your model artifacts
gcs_model_uri = os.environ.get("AIP_STORAGE_URI")
local_model_dir = "/model"
local_scaler_path = "/model/scaler.pkl"

if gcs_model_uri:
    client = storage.Client()
    bucket_name, prefix = gcs_model_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    # Clean out local model dir
    if os.path.isdir(local_model_dir):
        shutil.rmtree(local_model_dir)
    os.makedirs(local_model_dir, exist_ok=True)

    for blob in blobs:
        # skip “folders”
        if blob.name.endswith("/"):
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(local_model_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)

    # Additionally fetch scaler.pkl (assumes it's at gs://<bucket>/model/scaler.pkl)
    scaler_blob = bucket.blob("model/scaler.pkl")
    scaler_blob.download_to_filename(local_scaler_path)

else:
    print("AIP_STORAGE_URI not set, expecting /model baked into image")

# 3) Load model and scaler
model = tf.keras.models.load_model(local_model_dir)
scaler = joblib.load(local_scaler_path)

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json(force=True)
    instances = req.get("instances")
    if instances is None:
        return jsonify({"error": "Request JSON must have an 'instances' key."}), 400

    data = np.array(instances, dtype=np.float32)
    data_scaled = scaler.transform(data)

    preds = model.predict(data_scaled)

    if preds.ndim == 2 and preds.shape[1] > 1:
        result = np.argmax(preds, axis=1).tolist()
    else:
        result = preds.flatten().tolist()

    return jsonify({"predictions": result})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
