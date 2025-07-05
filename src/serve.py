# src/serve.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os
app = Flask(__name__)

# Where Vertex or your container runtime will mount the model
MODEL_PATH = os.environ.get("MODEL_DIR", "/opt/ml/model")
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON like: {"instances": [[f1, f2, ...], [f1, f2, ...], ...]}
    data = request.get_json(force=True, silent=True)
    if not data or "instances" not in data:
        return jsonify({"error": "Please provide JSON with an 'instances' key"}), 400

    instances = data["instances"]
    try:
        # Convert to Tensor (ensure dtype matches what your model expects)
        inputs = tf.convert_to_tensor(instances, dtype=tf.float32)
        preds = model.predict(inputs)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    return jsonify({"predictions": preds.tolist()}), 200

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
