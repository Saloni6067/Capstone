import numpy as np
import pandas as pd
import tensorflow as tf

# 1) Load your saved artifacts
model = tf.keras.models.load_model('model_output/tf_model')
scaler = pd.read_pickle('model_output/scaler.pkl')

# 2) Build a DataFrame for your new samples
#    (must have exactly the same columns & order as during training)
new_data = pd.DataFrame([
    {
        'latitude': 66.93,
        'longitude': 80.73,
        'brightness': 322.94,
        'bright_t31': 246.4,
        'frp': 3.11,     # will be capped below
        'scan': 0.68,
        'track': 0.74,
        'daynight': 'N',  # 'D' or 'N'
        'type': 0
    }
])

# 3) Apply the same preprocessing
new_data['frp'] = new_data['frp'].clip(upper=2000)
new_data['daynight'] = new_data['daynight'].map({'D': 0, 'N': 1})

# 4) Select & order your feature columns
feature_cols = ['latitude','longitude','brightness','bright_t31','frp',
                'scan','track','daynight','type']
X_new = new_data[feature_cols].values

# 5) Scale
X_new_scaled = scaler.transform(X_new)

# 6) Predict
probs = model.predict(X_new_scaled)   # shape: (n_samples, 3)
pred_idxs = np.argmax(probs, axis=1)  # class index 0,1,2

# 7) Map back to your labels ('h','l','n')
#    Use the same LabelEncoder from training — recreate it here:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(['h','l','n'])
pred_labels = le.inverse_transform(pred_idxs)

# 8) Show results
for i, (lbl, p) in enumerate(zip(pred_labels, probs)):
    print(f"Sample {i}: predicted={lbl}, softmax_probs={p}")
