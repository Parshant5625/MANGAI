from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.app.adapters.satellite.fusion_pipeline import LiveSatelliteFusionPipeline
from backend.app.core.errors import DataUnavailableError
from ml.common.external_contracts import validate_external_batch
from ml.common.provenance import DataBatch, DataProvenance


class FakeSentinel:
    def __init__(self, records):
        self.records = records

    def search_scenes(self, site_id, start, end, limit=10):
        return DataBatch(self.records, _provenance("satellite", len(self.records)))


class FakeLandsat:
    def __init__(self, scenes):
        self.scenes = scenes
        self.calls = []

    def discover(self, latitude, longitude, start_date=None, end_date=None):
        self.calls.append((latitude, longitude, start_date, end_date))
        return self.scenes


class FakeOptical:
    def ingest_scene(self, scene, site_id, *, aoi_bbox=None):
        return DataBatch(
            [{"scene_id": scene["scene_id"], "aoi_bbox": aoi_bbox}],
            _provenance("satellite", 1),
        )


class FakeFusion:
    def fuse(self, optical, thermal, *, aoi_bbox=None):
        return DataBatch(
            [
                {
                    "scene_id": optical.records[0]["scene_id"],
                    "thermal_scene_id": thermal["scene_id"],
                    "aoi_bbox": aoi_bbox,
                }
            ],
            _provenance("satellite", 1),
        )


def _provenance(dataset: str, rows: int) -> DataProvenance:
    now = datetime.now(UTC).isoformat()
    return DataProvenance(
        source_name="test",
        source_kind="satellite",
        mode="live",
        dataset=dataset,
        acquired_at=now,
        ingested_at=now,
        source_version="test",
        source_uri="test",
        checksum="test",
        license_note="test",
        quality_score=1.0,
        row_count=rows,
    )


def test_pipeline_discovers_matches_and_fuses(monkeypatch):
    sentinel = [{"scene_id": "s1", "datetime": "2026-08-10T00:00:00Z"}]
    landsat = [{"scene_id": "l1", "datetime": "2026-08-11T00:00:00Z", "assets": {"surface_temperature": "x"}}]
    provider = FakeLandsat(landsat)
    pipeline = LiveSatelliteFusionPipeline(FakeSentinel(sentinel), provider, FakeOptical(), FakeFusion())
    monkeypatch.setattr("backend.app.adapters.satellite.fusion_pipeline.validate_external_batch", lambda batch: [])

    result = pipeline.run(site_id="demo", start="2026-08-01", end="2026-08-31", latitude=20.0, longitude=76.0)

    assert result.sentinel_scene_count == 1
    assert result.fused_scene_count == 1
    assert result.thermal_scene_count == 1
    assert result.temporal_distance_days == (1.0,)
    assert result.batch.records[0]["thermal_scene_id"] == "l1"
    assert provider.calls == [(20.0, 76.0, "2026-07-16", "2026-09-16")]
    assert result.batch.records[0]["aoi_bbox"] == pytest.approx((75.98, 19.98, 76.02, 20.02))


def test_pipeline_fails_when_no_thermal_scene_exists():
    pipeline = LiveSatelliteFusionPipeline(
        FakeSentinel([{"scene_id": "s1", "datetime": "2026-08-10T00:00:00Z"}]),
        FakeLandsat([]),
        FakeOptical(),
        FakeFusion(),
    )
    with pytest.raises(DataUnavailableError, match="No Landsat ST scenes"):
        pipeline.run(site_id="demo", start="2026-08-01", end="2026-08-31", latitude=20.0, longitude=76.0)


def test_pipeline_fails_when_temporal_match_is_outside_window():
    pipeline = LiveSatelliteFusionPipeline(
        FakeSentinel([{"scene_id": "s1", "datetime": "2026-08-10T00:00:00Z"}]),
        FakeLandsat([{"scene_id": "l1", "datetime": "2026-08-30T00:00:00Z", "assets": {"surface_temperature": "x"}}]),
        FakeOptical(),
        FakeFusion(),
    )
    with pytest.raises(DataUnavailableError, match="No Sentinel-2 scene could be thermally fused") as exc_info:
        pipeline.run(
            site_id="demo",
            start="2026-08-01",
            end="2026-08-31",
            latitude=20.0,
            longitude=76.0,
            max_temporal_days=5,
        )

    assert exc_info.value.details["failures"][0]["error"] == "no temporal Landsat candidate"
