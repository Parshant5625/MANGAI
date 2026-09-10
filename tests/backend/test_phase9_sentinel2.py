from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.core.errors import DataUnavailableError


def _settings(**overrides):
    values = {
        "sentinel2_stac_url": "https://example.test/v1",
        "sentinel2_collection": "sentinel-2-l2a",
        "sentinel2_max_cloud_cover": 20.0,
        "sentinel2_bbox_delta": 0.05,
        "sentinel2_timeout_seconds": 5.0,
        "sentinel2_latitude": 21.8,
        "sentinel2_longitude": 80.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_live_sentinel2_requires_coordinates(monkeypatch) -> None:
    from backend.app.adapters.satellite.sentinel2 import Sentinel2STACProvider

    monkeypatch.setattr(
        "backend.app.adapters.satellite.sentinel2.get_settings",
        lambda: _settings(sentinel2_latitude=None, sentinel2_longitude=None),
    )
    with pytest.raises(DataUnavailableError, match="SENTINEL2_LATITUDE"):
        Sentinel2STACProvider()


def test_sentinel2_batch_has_live_provenance(monkeypatch) -> None:
    from backend.app.adapters.satellite.sentinel2 import Sentinel2STACProvider

    monkeypatch.setattr("backend.app.adapters.satellite.sentinel2.get_settings", lambda: _settings())
    provider = Sentinel2STACProvider()
    payload = {
        "features": [
            {
                "id": "S2A_TEST",
                "collection": "sentinel-2-l2a",
                "properties": {"datetime": "2026-09-01T10:00:00Z", "eo:cloud_cover": 8.0},
                "geometry": {"type": "Point", "coordinates": [80.0, 21.8]},
                "assets": {"B04": {"href": "https://example.test/B04.tif"}},
            }
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            import json

            return json.dumps(payload).encode()

    monkeypatch.setattr("backend.app.adapters.satellite.sentinel2.urlopen", lambda *args, **kwargs: Response())
    batch = provider.search_scenes("mine-1", "2026-09-01", "2026-09-02")

    assert len(batch.records) == 1
    assert batch.records[0]["scene_id"] == "S2A_TEST"
    assert batch.records[0]["assets"]["B04"].endswith("B04.tif")
    assert batch.provenance.mode == "live"
    assert batch.provenance.source_kind == "satellite"
    assert batch.provenance.dataset == "satellite_features"
    assert batch.provenance.row_count == 1
    assert batch.provenance.quality_score == pytest.approx(0.92)


def test_sentinel2_empty_result_is_data_unavailable(monkeypatch) -> None:
    from backend.app.adapters.satellite.sentinel2 import Sentinel2STACProvider

    monkeypatch.setattr("backend.app.adapters.satellite.sentinel2.get_settings", lambda: _settings())

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"features": []}'

    monkeypatch.setattr("backend.app.adapters.satellite.sentinel2.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(DataUnavailableError, match="No Sentinel-2"):
        Sentinel2STACProvider().search_scenes("mine-1", "2026-09-01", "2026-09-02")
