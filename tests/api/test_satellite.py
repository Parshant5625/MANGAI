from fastapi.testclient import TestClient

from backend.app.main import app


def test_satellite_scenes_contract() -> None:
    response = TestClient(app).get("/api/v1/satellite/scenes?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] <= 5
    assert isinstance(body["scenes"], list)
    assert body["data_mode"] == "DEMO / LOCAL FILE"
