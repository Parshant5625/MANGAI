"""Manual Phase 3 gate: execute real API inference and print evidence."""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from backend.app.main import app

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
print("POST /api/v1/predictions/reserve ->", response.status_code)
payload = response.json()
print(json.dumps({k: payload[k] for k in ("data_mode", "synthetic_data", "model_version")}, indent=2))
prediction = payload["prediction"]
print(
    "prediction:",
    {
        "probability": prediction["probability"],
        "prospectivity_class": prediction["prospectivity_class"],
        "predicted_grade_pct": prediction["predicted_grade_pct"],
        "predicted_thickness_m": prediction["predicted_thickness_m"],
        "confidence": prediction["confidence"],
    },
)
print(
    "resource_potential:",
    {
        k: prediction["resource_potential"][k]
        for k in ("label", "expected_tonnage", "p10", "p50", "p90")
    },
)
print("data_support:", json.dumps(prediction["data_support"], indent=2))
assert 0 <= prediction["probability"] <= 1
assert prediction["resource_potential"]["p10"] <= prediction["resource_potential"]["p90"]
assert prediction["data_support"]["model_version"] == payload["model_version"]

models_response = client.get("/api/v1/models")
print("GET /api/v1/models ->", models_response.status_code)
for model in models_response.json()["models"]:
    if str(model.get("model_name", "")).startswith("reserve"):
        print(
            {
                "model_name": model["model_name"],
                "version": model["version"],
                "algorithm": model["algorithm"],
                "status": model["status"],
                "roc_auc": model.get("metrics", {}).get("roc_auc"),
                "rmse": model.get("metrics", {}).get("rmse"),
            }
        )
print("API INFERENCE VERIFICATION OK")
sys.exit(0)
