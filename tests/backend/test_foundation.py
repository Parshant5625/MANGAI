from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import app


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _reserve_payload() -> dict[str, object]:
    return {
        "latitude": 21.4,
        "longitude": 80.3,
        "elevation_m": 640,
        "slope_deg": 12,
        "aspect_deg": 180,
        "depth_m": 28,
        "formation": "Manganiferous_Formation",
    }


def _assert_error_format(payload: dict) -> None:
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "details"}
    assert isinstance(payload["error"]["code"], str)
    assert isinstance(payload["error"]["message"], str)
    assert isinstance(payload["error"]["details"], dict)


def test_health_liveness_check(demo_store):
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "mangai-api"}


def test_ready_reports_structured_dependency_status(demo_store):
    get_settings.cache_clear()
    response = _client().get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "degraded"}
    assert payload["database"] is True
    assert payload["data_mode"] == "demo"
    assert payload["demo_data"] is True
    assert set(payload["models"]) == {"reserve_prospectivity", "production_forecast"}


def test_malformed_request_returns_error_format(demo_store):
    response = _client().post(
        "/api/v1/predictions/production",
        content="{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_invalid_probability_returns_validation_error(demo_store):
    response = _client().get("/api/v1/reserves/prospectivity?min_probability=1.25")

    assert response.status_code == 422
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_invalid_horizon_returns_validation_error(demo_store):
    response = _client().get("/api/v1/production/forecast?horizon=31")

    assert response.status_code == 422
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_malformed_bbox_returns_domain_validation_error(demo_store):
    response = _client().get("/api/v1/reserves/prospectivity?bbox=80.5,21.2,80.1,21.6")

    assert response.status_code == 422
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "INVALID_BBOX"


def test_missing_reserve_model_in_live_mode_returns_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_MODE", "live")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()

    try:
        response = _client().post("/api/v1/predictions/reserve", json=_reserve_payload())
    finally:
        get_settings.cache_clear()

    assert response.status_code == 503
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_missing_production_model_in_live_mode_returns_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_MODE", "live")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()

    try:
        response = _client().get("/api/v1/production/forecast?horizon=7")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 503
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_ready_reports_database_unavailable(monkeypatch, demo_store):
    monkeypatch.setattr("backend.app.main.is_database_available", lambda: False)

    response = _client().get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["database"] is False


def test_demo_mode_preserves_synthetic_api_behavior(demo_store):
    get_settings.cache_clear()
    response = _client().get("/api/v1/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_mode"] == "demo"
    assert payload["synthetic_data"] is True


def test_invalid_site_id_returns_not_found_error(demo_store):
    response = _client().get("/api/v1/overview?site_id=unknown-site")

    assert response.status_code == 404
    _assert_error_format(response.json())
    assert response.json()["error"]["code"] == "SITE_NOT_FOUND"
