import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartFeatures, PredictionResponse

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "heart_model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = metadata["features"]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Serves predictions from a Random Forest trained on the Heart Disease dataset.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model_type": metadata["model_type"],
        "features": FEATURES,
        "n_features": len(FEATURES),
        "test_accuracy": metadata["test_accuracy"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0][1])
    return PredictionResponse(heart_disease=proba >= 0.5, probability=round(proba, 4))
