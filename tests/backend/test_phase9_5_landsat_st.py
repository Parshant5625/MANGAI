from __future__ import annotations

import json

from backend.app.adapters.satellite.landsat_st import LandsatSTConfig, LandsatSurfaceTemperatureProvider


def test_selects_surface_temperature_asset():
    assets = {"ST_B10": {"href": "https://example.test/st.tif"}, "thumbnail": {"href": "x"}}
    assert LandsatSurfaceTemperatureProvider._select_st_asset(assets) == "https://example.test/st.tif"


def test_discovery_filters_cloud_cover(monkeypatch):
    provider = LandsatSurfaceTemperatureProvider(LandsatSTConfig(max_cloud_cover=20, max_items=5))
    payload = {
        "features": [
            {
                "id": "clear",
                "datetime": "2026-07-30T00:00:00Z",
                "properties": {"eo:cloud_cover": 10},
                "assets": {"ST": {"href": "clear.tif"}},
                "self": "clear-item",
            },
            {
                "id": "cloudy",
                "datetime": "2026-07-29T00:00:00Z",
                "properties": {"eo:cloud_cover": 80},
                "assets": {"ST": {"href": "cloudy.tif"}},
            },
        ]
    }
    monkeypatch.setattr(provider, "_get_json", lambda url: payload)

    scenes = provider.discover(20.0, 80.0, "2026-07-01", "2026-07-31")

    assert [scene["scene_id"] for scene in scenes] == ["clear"]
    assert scenes[0]["assets"]["surface_temperature"] == "clear.tif"


def test_discovery_rejects_invalid_date_window():
    provider = LandsatSurfaceTemperatureProvider()
    try:
        provider.discover(20.0, 80.0, "2026-08-01", "2026-07-01")
    except Exception as exc:
        assert "after" in str(exc).lower()
    else:
        raise AssertionError("invalid date window must be rejected")
