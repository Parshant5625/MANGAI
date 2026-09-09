"""Phase 4 API tests: grid endpoint, model compare, enhanced prediction metadata."""

from fastapi.testclient import TestClient

from backend.app.main import app


def test_reserve_grid_endpoint(demo_store):
    client = TestClient(app)
    response = client.get("/api/v1/reserves/grid?cells=4")
    assert response.status_code == 200
    payload = response.json()
    assert payload["synthetic_data"] is True
    assert payload["cells_per_side"] == 4
    assert len(payload["cells"]) == 16
    assert "prospectivity_ensemble" in payload["model_versions"]


def test_reserve_prediction_includes_intervals_and_support(demo_store):
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
    assert payload["synthetic_data"] is True
    cell = payload["prediction"]
    assert 0 <= cell["probability"] <= 1
    # Phase 4 enrichment present when artifacts exist.
    assert "grade_interval" in cell
    assert "thickness_interval" in cell
    assert "extrapolation" in cell
    if cell["grade_interval"] is not None:
        assert cell["grade_interval"]["lower"] <= cell["grade_interval"]["upper"]
    if cell["thickness_interval"] is not None:
        assert cell["thickness_interval"]["lower"] >= 0  # non-negative


def test_models_compare_endpoint(demo_store):
    client = TestClient(app)
    response = client.get("/api/v1/models/compare")
    assert response.status_code == 200
    comparison = response.json()["comparison"]
    assert comparison, "registry should contain trained reserve models"
    assert all("leakage_check_passed" in entry for entry in comparison)
