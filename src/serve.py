# src/serve.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np

app = Flask(__name__)

# Vertex AI mounts your SavedModel here by default:
MODEL_PATH = os.environ.get("AIP_MODEL_DIR", "/model")
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON {"instances": [[feat1, feat2, ..., featN], ...]}
    req = request.get_json(force=True)
    instances = req.get("instances")
    if instances is None:
        return jsonify({"error": "Request JSON must have an 'instances' key."}), 400

    # Convert to NumPy array of shape (batch_size, num_features)
    data = np.array(instances, dtype=np.float32)

    # Run inference
    preds = model.predict(data)

    # If it's a classification model with softmax, return class indices:
    if preds.ndim == 2 and preds.shape[1] > 1:
        result = np.argmax(preds, axis=1).tolist()
    else:
        # For regression or binary (1‐dim) outputs
        result = preds.flatten().tolist()

    return jsonify({"predictions": result})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    # For local testing: override AIP_MODEL_DIR to point to your saved_model folder
    app.run(host="0.0.0.0", port=8080)
