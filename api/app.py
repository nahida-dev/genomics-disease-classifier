# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

from src.config import RAW_DATA_DIR
from src.preprocessing import load_data, normalize_data
import pandas as pd

# ------------------------
# Paths
# ------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = RAW_DATA_DIR / "GSE15852_series_matrix.txt"
SELECTED_FEATURES_FILE = MODEL_DIR / "selected_200_features.csv"

# ------------------------
# Load artifacts
# ------------------------
model = joblib.load(MODEL_DIR / "model.pkl")
selector = joblib.load(MODEL_DIR / "selector.pkl")


def load_reference_samples():
    df = normalize_data(load_data(DATA_FILE))
    return df.T


reference_samples = load_reference_samples()

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Genomics Disease Classifier API")


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/selected-features")
def selected_features():
    if not SELECTED_FEATURES_FILE.exists():
        return {
            "error": (
                "selected_200_features.csv not found. Run training first with "
                "`python -m src.train`."
            )
        }

    df = pd.read_csv(SELECTED_FEATURES_FILE)
    return {
        "feature_count": int(len(df)),
        "features": df.to_dict(orient="records"),
    }


@app.get("/sample-input")
def sample_input(sample_index: int = 0):
    if sample_index < 0 or sample_index >= len(reference_samples):
        return {
            "error": (
                f"sample_index must be between 0 and {len(reference_samples) - 1}"
            )
        }

    sample = reference_samples.iloc[sample_index]

    return {"features": sample.astype(float).tolist()}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        x = np.array(request.features).reshape(1, -1)

        # Validate input size
        if x.shape[1] != selector.n_features_in_:
            return {
                "error": f"Expected {selector.n_features_in_} features, got {x.shape[1]}"
            }

        x_selected = selector.transform(x)

        pred = model.predict(x_selected)[0]
        prob = model.predict_proba(x_selected)[0, 1]

        return {
            "prediction": int(pred),
            "label": "Cancer" if pred == 1 else "Normal",
            "cancer_probability": float(prob),
        }

    except Exception as e:
        return {"error": str(e)}
