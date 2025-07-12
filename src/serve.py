# src/serve.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
from google.cloud import storage
import shutil

app = Flask(__name__)

# 1) Determine where Vertex AI has staged your model artifacts
gcs_model_uri = os.environ.get("AIP_STORAGE_URI")
local_model_dir = "/model"

if gcs_model_uri:
    # 2) If it’s a GCS path, pull it down into /model
    client = storage.Client()
    bucket_name, prefix = gcs_model_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    # clean out any existing local
    if os.path.isdir(local_model_dir):
        shutil.rmtree(local_model_dir)
    os.makedirs(local_model_dir, exist_ok=True)
    for blob in blobs:
        # skip “folders”
        if blob.name.endswith("/"):
            continue
        # write each file under /model/<relative path>
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(local_model_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
else:
    # If no AIP_STORAGE_URI, assume your Docker IMAGE baked in a /model folder
    print("⚠️  AIP_STORAGE_URI not set, expecting /model baked into image")

# 3) Finally load the model
model = tf.keras.models.load_model(local_model_dir)

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json(force=True)
    instances = req.get("instances")
    if instances is None:
        return jsonify({"error": "Request JSON must have an 'instances' key."}), 400

    data = np.array(instances, dtype=np.float32)
    preds = model.predict(data)

    if preds.ndim == 2 and preds.shape[1] > 1:
        result = np.argmax(preds, axis=1).tolist()
    else:
        result = preds.flatten().tolist()

    return jsonify({"predictions": result})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    # local debug
    app.run(host="0.0.0.0", port=8080)
