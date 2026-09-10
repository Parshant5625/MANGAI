from __future__ import annotations

import pandas as pd
import pytest

from ml.common.provenance import DataBatch, DataProvenance
from ml.reserve.live_fusion import fuse_live_satellite_with_geology


def _batch(*, mode: str = "live") -> DataBatch:
    records = [
        {
            "sample_id": "S-PIX-1",
            "latitude": 20.0001,
            "longitude": 80.0001,
            "blue_b2": 0.10,
            "green_b3": 0.12,
            "red_b4": 0.14,
            "nir_b8": 0.40,
            "swir_b11": 0.30,
            "swir_b12": 0.24,
            "land_surface_temperature": 31.0,
        },
        {
            "sample_id": "S-PIX-2",
            "latitude": 20.02,
            "longitude": 80.02,
            "blue_b2": 0.11,
            "green_b3": 0.13,
            "red_b4": 0.15,
            "nir_b8": 0.38,
            "swir_b11": 0.29,
            "swir_b12": 0.23,
            "land_surface_temperature": 30.0,
        },
    ]
    provenance = DataProvenance(
        source_name="Sentinel-2 + Landsat",
        source_kind="satellite",
        mode=mode,
        dataset="satellite_features",
        acquired_at="2026-09-01T00:00:00Z",
        ingested_at="2026-09-01T01:00:00Z",
        source_version="test",
        source_uri="test://satellite",
        checksum="abc123",
        license_note="test",
        quality_score=1.0,
        row_count=len(records),
    )
    return DataBatch(records=records, provenance=provenance)


def _geology() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "G-1",
                "latitude": 20.0000,
                "longitude": 80.0000,
                "elevation_m": 400.0,
                "slope_deg": 12.0,
                "aspect_deg": 90.0,
                "depth_m": 25.0,
                "formation": "Gondite",
                "mn_pct": 42.0,
                "ore_thickness_m": 4.0,
            }
        ]
    )


def test_live_fusion_uses_nearest_context_and_does_not_copy_targets() -> None:
    result = fuse_live_satellite_with_geology(_batch(), _geology(), max_distance_m=100)

    assert result.matched == 1
    assert result.unmatched_satellite == 1
    assert result.match_rate == 0.5
    assert result.data.iloc[0]["formation"] == "Gondite"
    assert result.data.iloc[0]["elevation_m"] == 400.0
    assert "mn_pct" not in result.data.columns
    assert "ore_thickness_m" not in result.data.columns
    assert result.data.iloc[0]["geology_match_distance_m"] < 100


def test_live_fusion_rejects_demo_satellite_batch() -> None:
    with pytest.raises(ValueError, match="live satellite batch"):
        fuse_live_satellite_with_geology(_batch(mode="demo"), _geology())


def test_live_fusion_rejects_missing_context_columns() -> None:
    geology = _geology().drop(columns=["formation"])
    with pytest.raises(ValueError, match="missing required columns"):
        fuse_live_satellite_with_geology(_batch(), geology)
