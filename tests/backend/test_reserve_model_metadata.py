"""API verification: reserve predictions must carry real model metadata.

Guards against hard-coded predictions/versions: the /predictions/reserve
response version must come from the served artifact's training metadata
(falling back to the labelled heuristic version only when no artifact exists).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app

_LEGACY_HARD_CODED_VERSION = "reserve-xgb-2026.09.001"


def test_reserve_prediction_model_version_is_dynamic(demo_store):
    client = TestClient(app)
    response = client.post(
        "/api/v1/predictions/reserve",
        json={
            "latitude": 21.4,
            "longitude": 80.3,
            "elevation_m": 640,
            "slope_deg": 12,
            "aspect_deg": 180,
            "depth_m": 28,
            "formation": "Manganiferous_Formation",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] != _LEGACY_HARD_CODED_VERSION
    assert payload["synthetic_data"] is True
    data_support = payload["prediction"]["data_support"]
    assert data_support["model_version"] == payload["model_version"]
    assert data_support["prediction_type"] == "manganese_prospectivity"


def test_model_registry_lists_reserve_models(demo_store):
    client = TestClient(app)
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    names = {model["model_name"] for model in response.json()["models"]}
    assert "reserve_prospectivity" in names
