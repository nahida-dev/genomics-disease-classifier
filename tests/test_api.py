from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import api.app as api_app


client = TestClient(api_app.app)


def test_predict_happy_path():
    sample_response = client.get("/sample-input", params={"sample_index": 0})
    assert sample_response.status_code == 200

    payload = sample_response.json()
    assert "features" in payload
    assert len(payload["features"]) == api_app.selector.n_features_in_

    predict_response = client.post("/predict", json=payload)
    assert predict_response.status_code == 200

    body = predict_response.json()
    assert set(body.keys()) == {"prediction", "label", "cancer_probability"}
    assert body["prediction"] in {0, 1}
    assert body["label"] in {"Cancer", "Normal"}
    assert 0.0 <= body["cancer_probability"] <= 1.0


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert response.status_code == 200

    body = response.json()
    assert "error" in body
    assert str(api_app.selector.n_features_in_) in body["error"]


def test_selected_features_returns_saved_file_contents():
    response = client.get("/selected-features")
    assert response.status_code == 200

    body = response.json()
    assert "feature_count" in body
    assert "features" in body
    assert body["feature_count"] == len(body["features"])

    if body["features"]:
        assert {"rank", "feature_name", "f_score"} <= set(body["features"][0].keys())


def test_selected_features_handles_missing_file(monkeypatch, tmp_path):
    missing_file = tmp_path / "selected_200_features.csv"
    monkeypatch.setattr(api_app, "SELECTED_FEATURES_FILE", missing_file)

    response = client.get("/selected-features")
    assert response.status_code == 200

    body = response.json()
    assert "error" in body
    assert "selected_200_features.csv not found" in body["error"]
