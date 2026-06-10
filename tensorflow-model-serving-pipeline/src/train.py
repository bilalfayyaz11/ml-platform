import os
import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TrainingPipeline:
    def load_data(self) -> tuple:
        iris = load_iris()
        X = iris.data.astype("float32")
        y = iris.target.astype("int64")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype("float32")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.scaler = scaler
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names.tolist()

        return X_train, X_test, y_train, y_test

    def build_model(self, input_dim: int, num_classes: int) -> object:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, model: object, X_train, y_train) -> dict:
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=8,
            verbose=2,
        )

        return {
            "accuracy": history.history.get("accuracy", []),
            "val_accuracy": history.history.get("val_accuracy", []),
        }

    def save_artifacts(self, model: object, scaler: object, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "iris_classifier.keras")
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        metadata_path = os.path.join(output_dir, "model_metadata.json")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "model_file": "iris_classifier.keras",
            "scaler_file": "scaler.joblib",
            "framework": "TensorFlow",
            "dataset": "sklearn iris",
            "input_dim": 4,
            "num_classes": 3,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        for path in [model_path, scaler_path, metadata_path]:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise IOError(f"Artifact save failed or zero-byte file detected: {path}")


def main():
    output_dir = os.environ.get("MODEL_OUTPUT_DIR", "/app/models")

    pipeline = TrainingPipeline()
    X_train, X_test, y_train, y_test = pipeline.load_data()

    model = pipeline.build_model(input_dim=X_train.shape[1], num_classes=3)
    history = pipeline.train(model, X_train, y_train)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    pipeline.save_artifacts(model, pipeline.scaler, output_dir)

    print("Training completed successfully")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model artifacts saved to: {output_dir}")
    print(f"History keys: {list(history.keys())}")


if __name__ == "__main__":
    main()
