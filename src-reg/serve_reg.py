#!/usr/bin/env python
import os
import pickle
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.cloud import storage

app = Flask(__name__)

# Environment-configured bucket & prefix
GCS_BUCKET = os.environ['GCS_BUCKET']
GCS_PREFIX = os.environ.get('GCS_PREFIX', 'RegModel')
MODEL_PATH = '/model/tf_model'

# 1) Download every blob under {GCS_PREFIX}/tf_model/, recreating folders:
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)
prefix_path = f"{GCS_PREFIX}/tf_model/"
for blob in bucket.list_blobs(prefix=prefix_path):
    if blob.name.endswith('/'):
        continue
    rel_path = blob.name[len(prefix_path):]                # e.g. "variables/variables.index"
    local_file = os.path.join(MODEL_PATH, rel_path)
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    blob.download_to_filename(local_file)

# 2) Load SavedModel directory
model = keras.models.load_model(MODEL_PATH)

# 3) Load scaler/preprocessor
with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb') as f:
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
