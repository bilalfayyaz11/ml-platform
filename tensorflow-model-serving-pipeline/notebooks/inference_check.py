import os
import joblib
import numpy as np
import tensorflow as tf

model_path = "/app/models/iris_classifier.keras"
scaler_path = "/app/models/scaler.joblib"

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype="float32")
scaled = scaler.transform(sample)
prediction = model.predict(scaled, verbose=0)

print("Prediction probabilities:", prediction.tolist())
print("Predicted class:", int(np.argmax(prediction[0])))
