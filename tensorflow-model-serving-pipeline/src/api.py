import os
import json
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from db import PredictionLogger


class PredictionAPI:
    def __init__(self):
        self.model_dir = os.environ.get("MODEL_DIR", "/app/models")
        self.model_path = os.path.join(self.model_dir, "iris_classifier.keras")
        self.scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.logger = PredictionLogger()

        self.load_artifacts()
        self.connect_database()

    def load_artifacts(self):
        missing = [
            path for path in [self.model_path, self.scaler_path, self.metadata_path]
            if not os.path.exists(path) or os.path.getsize(path) == 0
        ]

        if missing:
            raise FileNotFoundError(f"Required model artifacts missing: {missing}")

        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def connect_database(self):
        dsn = os.environ.get("DATABASE_DSN")
        if not dsn:
            raise ConnectionError("DATABASE_DSN environment variable is required")
        self.logger.connect(dsn)

    def health(self) -> dict:
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
        }

    def predict(self, payload: dict) -> dict:
        features = payload.get("features")

        if not isinstance(features, list) or len(features) != 4:
            raise ValueError("Payload must contain {'features': list[float]} with exactly 4 values")

        feature_array = np.array([features], dtype="float32")
        scaled_features = self.scaler.transform(feature_array)

        probabilities = self.model.predict(scaled_features, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        logged_id = self.logger.log_prediction(features, prediction, confidence)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "logged_id": logged_id,
        }

    def model_info(self) -> dict:
        return {
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "metadata": self.metadata,
        }


app = Flask(__name__)
prediction_api = PredictionAPI()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(prediction_api.health()), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        result = prediction_api.predict(request.get_json(force=True))
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify(prediction_api.model_info()), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
