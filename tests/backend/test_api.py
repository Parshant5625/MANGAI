from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_and_core_endpoints(demo_store):
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    overview = client.get("/api/v1/overview")
    assert overview.status_code == 200
    payload = overview.json()
    assert payload["synthetic_data"] is True
    assert "resource_potential_tonnage" in payload

    prospectivity = client.get("/api/v1/reserves/prospectivity?limit=20")
    assert prospectivity.status_code == 200
    assert prospectivity.json()["count"] >= 1

    forecast = client.get("/api/v1/production/forecast?horizon=7")
    assert forecast.status_code == 200
    body = forecast.json()
    assert body["horizon_days"] == 7
    assert "shortfall_probability" in body

    recs = client.get("/api/v1/recommendations")
    assert recs.status_code == 200
    assert "recommendations" in recs.json()

    quality = client.get("/api/v1/data-quality")
    assert quality.status_code == 200
    assert quality.json()["overall_score"] >= 0
