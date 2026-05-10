# api/app.py

import logging
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status
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
logger = logging.getLogger(__name__)

MODEL_FILE = MODEL_DIR / "model.pkl"
SELECTOR_FILE = MODEL_DIR / "selector.pkl"

model = None
selector = None
reference_samples = None


def _load_joblib_artifact(file_path: Path, artifact_name: str):
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{artifact_name} not found. Run training first with "
                "`python -m src.train`."
            ),
        )

    try:
        return joblib.load(file_path)
    except Exception as exc:
        logger.exception("Failed to load %s from %s", artifact_name, file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load {artifact_name}.",
        ) from exc


def get_model():
    global model

    if model is None:
        model = _load_joblib_artifact(MODEL_FILE, "model artifact")

    return model


def get_selector():
    global selector

    if selector is None:
        selector = _load_joblib_artifact(SELECTOR_FILE, "selector artifact")

    return selector


def load_reference_samples():
    if not DATA_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Reference data file not found. Ensure "
                "`data/raw/GSE15852_series_matrix.txt` exists."
            ),
        )

    try:
        df = normalize_data(load_data(DATA_FILE))
    except Exception as exc:
        logger.exception("Failed to load reference samples from %s", DATA_FILE)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load reference samples.",
        ) from exc

    return df.T


def get_reference_samples():
    global reference_samples

    if reference_samples is None:
        reference_samples = load_reference_samples()

    return reference_samples

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
    missing_files = [
        str(path.relative_to(BASE_DIR))
        for path in (MODEL_FILE, SELECTOR_FILE, DATA_FILE)
        if not path.exists()
    ]
    return {
        "status": "ok" if not missing_files else "degraded",
        "missing_files": missing_files,
    }


@app.get("/selected-features")
def selected_features():
    if not SELECTED_FEATURES_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "selected_200_features.csv not found. Run training first with "
                "`python -m src.train`."
            ),
        )

    df = pd.read_csv(SELECTED_FEATURES_FILE)
    return {
        "feature_count": int(len(df)),
        "features": df.to_dict(orient="records"),
    }


@app.get("/sample-input")
def sample_input(sample_index: int = 0):
    samples = get_reference_samples()

    if sample_index < 0 or sample_index >= len(samples):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"sample_index must be between 0 and {len(samples) - 1}",
        )

    sample = samples.iloc[sample_index]

    return {"features": sample.astype(float).tolist()}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        current_selector = get_selector()
        current_model = get_model()
        x = np.array(request.features).reshape(1, -1)

        # Validate input size
        if x.shape[1] != current_selector.n_features_in_:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Expected {current_selector.n_features_in_} features, "
                    f"got {x.shape[1]}"
                ),
            )

        x_selected = current_selector.transform(x)

        pred = current_model.predict(x_selected)[0]
        prob = current_model.predict_proba(x_selected)[0, 1]

        return {
            "prediction": int(pred),
            "label": "Cancer" if pred == 1 else "Normal",
            "cancer_probability": float(prob),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc
