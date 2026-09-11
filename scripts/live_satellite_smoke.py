"""Run a fail-closed end-to-end satellite fusion smoke test.

The smoke test uses real Sentinel-2 and Landsat Collection 2 assets from
Microsoft Planetary Computer. It does not use demo satellite data, synthetic
fallbacks, or model training.

Required environment variables:
    SENTINEL2_LATITUDE
    SENTINEL2_LONGITUDE

Optional:
    LIVE_SMOKE_SITE_ID (default: smoke-test-site)
    LIVE_SMOKE_START (default: 30 days before LIVE_SMOKE_END)
    LIVE_SMOKE_END (default: today UTC)
    SENTINEL2_MAX_CLOUD_COVER
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import numpy as np

from backend.app.adapters.satellite.fusion_pipeline import LiveSatelliteFusionPipeline
from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError


def _date_range() -> tuple[str, str]:
    end = os.getenv("LIVE_SMOKE_END", datetime.now(UTC).date().isoformat())
    start = os.getenv(
        "LIVE_SMOKE_START",
        (datetime.fromisoformat(end).date() - timedelta(days=30)).isoformat(),
    )
    return start, end


def main() -> int:
    settings = get_settings()
    latitude = settings.sentinel2_latitude
    longitude = settings.sentinel2_longitude
    site_id = os.getenv("LIVE_SMOKE_SITE_ID", "smoke-test-site")
    start, end = _date_range()

    print("MANGAI live satellite fusion smoke test")
    print("provider: Microsoft Planetary Computer")
    print(f"window: {start} -> {end}")
    print(f"site_id: {site_id}")
    print(f"coordinates: {latitude}, {longitude}")

    if latitude is None or longitude is None:
        print("BLOCKED: set SENTINEL2_LATITUDE and SENTINEL2_LONGITUDE first.")
        return 2

    try:
        result = LiveSatelliteFusionPipeline().run(
            site_id=site_id,
            start=start,
            end=end,
            latitude=latitude,
            longitude=longitude,
            max_temporal_days=16,
            limit=2,
        )
    except DataUnavailableError as exc:
        print(f"FAIL: live satellite fusion unavailable: {exc.message}")
        if exc.details:
            print(f"details: {exc.details}")
        return 1
    except Exception as exc:
        print(f"FAIL: unexpected fusion error: {type(exc).__name__}: {exc}")
        return 1

    records = result.batch.records
    thermal = np.asarray(
        [record.get("land_surface_temperature") for record in records],
        dtype=object,
    )
    thermal_valid = np.array([value is not None and np.isfinite(float(value)) for value in thermal], dtype=bool)

    print(f"PASS: Sentinel-2 scenes: {result.sentinel_scene_count}")
    print(f"PASS: Landsat thermal scenes discovered: {result.thermal_scene_count}")
    print(f"PASS: fused Sentinel-2 scenes: {result.fused_scene_count}")
    print(f"PASS: fused satellite pixels: {len(records)}")
    print(f"thermal coverage: {thermal_valid.mean() if len(thermal_valid) else 0.0:.3f}")
    print(f"temporal distances (days): {list(result.temporal_distance_days)}")
    print(f"quality_score: {result.batch.provenance.quality_score:.3f}")
    print(f"source: {result.batch.provenance.source_name}")
    print(f"mode: {result.batch.provenance.mode}")
    print(f"dataset: {result.batch.provenance.dataset}")

    if not records or not thermal_valid.any():
        print("FAIL: fusion returned no usable thermal observations.")
        return 3

    sample = records[0]
    print(f"sample WGS84: lat={sample.get('latitude')}, lon={sample.get('longitude')}")
    print(f"sample NDVI: {sample.get('ndvi')}")
    print(f"sample LST C: {sample.get('land_surface_temperature')}")
    print("PASS: real Sentinel-2 + Landsat ST data reached the canonical satellite feature contract.")
    print("NEXT: run live reserve inference against the fused satellite features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
