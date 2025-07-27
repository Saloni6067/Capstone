# reg_serve.py
#!/usr/bin/env python
import os
import pickle
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
MODEL_PATH = os.environ.get('MODEL_PATH', '/model/tf_model')

model = keras.models.load_model(os.path.join(MODEL_PATH, 'saved_model.pb'))
with open(os.path.join(MODEL_PATH, 'preprocessor.pkl'), 'rb') as f:
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