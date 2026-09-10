from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.live_reserve import LiveSatelliteReserveRequest


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
