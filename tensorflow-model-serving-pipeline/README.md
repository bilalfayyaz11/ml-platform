# TensorFlow Model Serving Pipeline

## What This Does

This implementation provides a complete containerized machine learning pipeline for training, serving, and logging TensorFlow model predictions.

The system trains an Iris classification model, persists model artifacts to host-mounted storage, serves predictions through a Flask REST API, logs prediction records into PostgreSQL, and exposes a JupyterLab workspace for interactive model inspection.

The platform is designed around reproducibility, non-root container execution, multi-stage Docker builds, persistent artifact storage, health checks, and service orchestration through Docker Compose.

This type of architecture is used by MLOps, AI Platform, and ML Infrastructure teams to move machine learning workflows from local scripts into repeatable, containerized runtime environments.

## Architecture

    +--------------------------------------+
    | TensorFlow Training Container        |
    | src/train.py                         |
    | Iris Dataset                         |
    | StandardScaler                       |
    | Keras Model                          |
    +------------------+-------------------+
                       |
                       v
    +--------------------------------------+
    | Persistent Model Artifacts           |
    | models/iris_classifier.keras         |
    | models/scaler.joblib                 |
    | models/model_metadata.json           |
    +------------------+-------------------+
                       |
                       v
    +--------------------------------------+
    | Docker Compose ML Stack              |
    +------------------+-------------------+
                       |
        +--------------+---------------+
        |                              |
        v                              v

+---------------------------+    +---------------------------+
| Flask Prediction API      |    | PostgreSQL Backend        |
| Port 5000                 |--->| predictions table         |
| /health                   |    | prediction audit records  |
| /predict                  |    +---------------------------+
| /model/info               |
+---------------------------+
        |
        v
+---------------------------+
| JupyterLab Workspace      |
| Port 8888                 |
| notebook artifact access  |
+---------------------------+

## Prerequisites

- Ubuntu 24.04
- Docker Engine
- Docker Compose Plugin
- Git
- curl
- jq
- tree
- Python knowledge
- Basic machine learning concepts
- Sufficient disk space for TensorFlow image builds

## Setup & Installation

sudo apt-get update

sudo apt-get install -y ca-certificates curl gnupg git tree jq

sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

sudo chmod a+r /etc/apt/keyrings/docker.gpg

. /etc/os-release

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl enable docker

sudo systemctl start docker

sudo docker --version

sudo docker compose version

## How to Reproduce

Build the multi-stage ML image:

sudo docker build -t ml-pipeline:v1 .

Verify the image exists:

sudo docker images | grep ml-pipeline

Verify non-root runtime configuration:

sudo docker inspect ml-pipeline:v1 --format 'Configured user: {{.Config.User}}'

Run the training container with persistent artifact storage:

sudo docker run --rm \
  -e MODEL_OUTPUT_DIR=/app/models \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  ml-pipeline:v1

Verify model artifacts:

ls -lh models

test -s models/iris_classifier.keras

test -s models/scaler.joblib

test -s models/model_metadata.json

Start the full ML platform:

sudo docker compose up -d --build

Check service status:

sudo docker compose ps

Validate API health:

curl -s http://localhost:5000/health | jq .

Inspect model metadata:

curl -s http://localhost:5000/model/info | jq .

Send a prediction request:

curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}' | jq .

Verify PostgreSQL prediction logging:

sudo docker compose exec -T postgres psql -U mluser -d ml_predictions \
  -c "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5;"

Run notebook-style inference inside Jupyter:

sudo docker compose exec -T jupyter python /app/notebooks/inference_check.py

Stop the stack:

sudo docker compose down

## API Endpoints

GET /health

Returns application health and confirms whether the model is loaded.

GET /model/info

Returns model input shape, output shape, framework metadata, dataset details, and artifact information.

POST /predict

Accepts a feature vector and returns prediction, confidence score, and database logging ID.

Example payload:

{
  "features": [5.1, 3.5, 1.4, 0.2]
}

## Tools Used

- Docker Engine
- Docker Compose
- Python 3.11
- TensorFlow
- Keras
- Flask
- Gunicorn
- PostgreSQL
- psycopg2
- scikit-learn
- NumPy
- pandas
- Joblib
- JupyterLab
- Bash
- jq
- Linux

## Key Skills Demonstrated

- MLOps platform design
- Multi-stage Docker image builds
- TensorFlow model training
- Model artifact persistence
- REST model serving
- API health checks
- PostgreSQL prediction logging
- Docker Compose service orchestration
- JupyterLab integration
- Non-root container execution
- Runtime environment configuration
- ML artifact lifecycle management
- Prediction audit logging
- Containerized ML reproducibility
- AI platform engineering fundamentals

## Real-World Use Case

A machine learning platform team could use this architecture to standardize how models are trained, packaged, served, and monitored across development environments. The model training container produces persistent artifacts, the API service loads those artifacts at startup, PostgreSQL stores prediction history for auditability, and JupyterLab gives data scientists a live workspace for inspection and experimentation. This pattern is especially useful for early-stage MLOps systems where teams need reproducible model workflows before moving to Kubernetes, model registries, feature stores, or managed inference platforms.

## Lessons Learned

- ML containers need persistent artifact storage because containers themselves are disposable.
- Multi-stage builds keep runtime images cleaner by separating build dependencies from serving dependencies.
- Model-serving APIs should load artifacts at startup so missing or corrupted models fail fast.
- Prediction logging is important for traceability, debugging, monitoring, and later model evaluation.
- JupyterLab can share the same mounted artifacts as the serving API, making experimentation and production validation easier to connect.
- CPU-only TensorFlow environments may show CUDA, cuDNN, or TensorRT warnings, but those are expected when no GPU is attached.

## Troubleshooting Log

Issue:
The original Docker installation instructions used a hardcoded Ubuntu Jammy repository.

Resolution:
Used the active Ubuntu codename from /etc/os-release so Docker installs correctly on Ubuntu 24.04 Noble.

Issue:
Docker group refresh through newgrp can interrupt portal SSH sessions.

Resolution:
Used sudo docker commands throughout the workflow for reliable execution in temporary cloud terminals.

Issue:
The API must not start successfully without trained model artifacts.

Resolution:
Implemented startup artifact validation for the Keras model, scaler, and metadata file.

Issue:
Prediction records needed to prove the API was connected to PostgreSQL.

Resolution:
Added a PredictionLogger module that inserts each prediction into the predictions table and returns the generated row ID.

Issue:
The .env file must not be baked into an image layer.

Resolution:
Added .env to .dockerignore so Compose can use it locally without including it in the Docker build context.

Issue:
TensorFlow printed CUDA, cuDNN, and TensorRT warnings during inference.

Resolution:
Confirmed the environment is CPU-only and the warnings do not prevent successful model loading or prediction.
