from fastapi.testclient import TestClient

from backend.app.main import app


def test_reserve_prediction_and_simulation(demo_store):
    client = TestClient(app)
    reserve = client.post(
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
    assert reserve.status_code == 200
    cell = reserve.json()["prediction"]
    assert 0 < cell["probability"] < 1
    assert cell["resource_potential"]["p10"] <= cell["resource_potential"]["p90"]

    simulated = client.post(
        "/api/v1/recommendations/simulate",
        json={"reduce_downtime_pct": 20, "defer_weather_sensitive_blasts": True},
    )
    assert simulated.status_code == 200
    assert all(item["status"] == "SIMULATED" for item in simulated.json()["recommendations"])
