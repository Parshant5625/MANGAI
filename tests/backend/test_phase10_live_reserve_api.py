from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.live_reserve import LiveSatelliteReserveRequest, LiveSatelliteReserveResponse


def test_live_reserve_request_accepts_valid_window() -> None:
    request = LiveSatelliteReserveRequest(
        site_id="moil-site-1",
        start="2026-08-01",
        end="2026-08-31",
        latitude=20.0,
        longitude=80.0,
    )
    assert request.max_temporal_days == 16
    assert request.max_geology_distance_m == 500


def test_live_reserve_request_rejects_reversed_window() -> None:
    with pytest.raises(ValidationError, match="start must not be after end"):
        LiveSatelliteReserveRequest(
            site_id="moil-site-1",
            start="2026-09-01",
            end="2026-08-31",
            latitude=20.0,
            longitude=80.0,
        )


def test_live_reserve_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LiveSatelliteReserveRequest(
            site_id="moil-site-1",
            start="2026-08-01",
            end="2026-08-31",
            latitude=20.0,
            longitude=80.0,
            synthetic_fallback=True,
        )


def test_live_reserve_response_contract_accepts_service_payload() -> None:
    payload = {
        "data_mode": "live",
        "synthetic_data": True,
        "mixed_data": True,
        "boundary_notice": "prototype resource potential; not an official mineral resource or reserve",
        "site_id": "demo-site",
        "count": 1,
        "matched_geological_context": 1,
        "unmatched_satellite": 0,
        "geology_provenance": {"source_name": "synthetic_geological_demo"},
        "satellite_provenance": {"source_name": "Planetary Computer"},
        "sentinel_scene_count": 1,
        "thermal_scene_count": 1,
        "temporal_distance_days": [2.0],
        "cells": [
            {
                "id": "s1",
                "latitude": 20.0,
                "longitude": 80.0,
                "probability": 0.8,
                "prospectivity_class": "HIGH",
                "predicted_grade_pct": 20.0,
                "predicted_thickness_m": 2.0,
                "confidence": 0.9,
                "resource_potential": {
                    "label": "prototype resource potential",
                    "expected_tonnage": 100.0,
                    "p10": 50.0,
                    "p50": 100.0,
                    "p90": 150.0,
                },
                "data_support": {
                    "mode": "live",
                    "satellite_source": "Planetary Computer",
                    "satellite_provenance_checksum": "abc",
                    "geology_source": "synthetic_demo",
                    "geology_match_distance_m": 100.0,
                    "feature_completeness": 1.0,
                    "model_version": "2026.09.005",
                    "boundary": "prototype resource potential; not an official mineral resource or reserve",
                },
            }
        ],
    }
    response = LiveSatelliteReserveResponse.model_validate(payload)
    assert response.cells[0].resource_potential.p50 == 100.0
