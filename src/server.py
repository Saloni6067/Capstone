# Optional: src/serve.py (for Cloud Run)
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = 'model/tf_model'
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['instances']  # expect list of feature arrays
    arr = np.array(data)
    preds = model.predict(arr).tolist()
    return jsonify({'predictions': preds})

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))