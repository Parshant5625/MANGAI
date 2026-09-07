"""Phase 2 tests: geological/satellite fusion."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.reserve.fusion import fuse_reserve_datasets


def test_fusion_joins_all_samples_on_alignment_keys(demo_store):
    geo = demo_store.geological()
    sat = demo_store.satellite()
    result = fuse_reserve_datasets(geo, sat)
    assert result.matched == len(geo)
    assert result.merge_rate == pytest.approx(1.0)
    assert result.unmatched_geological == 0
    assert result.unmatched_satellite == 0


def test_fused_table_has_no_duplicated_alignment_columns(demo_store):
    fused = fuse_reserve_datasets(demo_store.geological(), demo_store.satellite()).data
    for key in ("sample_id", "latitude", "longitude"):
        assert key in fused.columns
        assert f"{key}_sat" not in fused.columns
    assert fused["sample_id"].is_unique


def test_fusion_fails_loudly_when_contracts_are_violated(demo_store):
    broken = demo_store.satellite().drop(columns=["nir_b8"])
    with pytest.raises(ValueError):
        fuse_reserve_datasets(demo_store.geological(), broken)


def _satellite_frame_for(geo: pd.DataFrame) -> pd.DataFrame:
    sat = geo[["sample_id", "latitude", "longitude"]].copy()
    sat["blue_b2"] = 0.1
    sat["green_b3"] = 0.1
    sat["red_b4"] = 0.1
    sat["nir_b8"] = 0.3
    sat["swir_b11"] = 0.2
    sat["swir_b12"] = 0.2
    sat["ndvi"] = 0.5
    sat["ndwi"] = -0.2
    sat["swir_ratio"] = 1.0
    sat["bare_soil_index"] = 0.0
    sat["land_surface_temperature"] = 31.0
    return sat


def test_fusion_reports_unmatched_satellite_rows(demo_store):
    geo = demo_store.geological().head(10)
    sat = _satellite_frame_for(geo)
    orphan = sat.iloc[[0]].copy()
    orphan["sample_id"] = "GS99999"
    sat_with_orphan = pd.concat([sat, orphan], ignore_index=True)

    result = fuse_reserve_datasets(geo, sat_with_orphan)
    assert result.matched == 10
    assert result.unmatched_satellite == 1
    assert result.merge_rate == pytest.approx(1.0)


def test_fusion_preserves_both_geological_and_satellite_columns(demo_store):
    fused = fuse_reserve_datasets(demo_store.geological(), demo_store.satellite()).data
    for column in ("mn_pct", "is_manganese", "nir_b8", "ndvi", "land_surface_temperature"):
        assert column in fused.columns


def test_outer_fusion_keeps_unmatched_geological_rows(demo_store):
    geo = demo_store.geological().head(10)
    sat = _satellite_frame_for(geo).head(5)
    result = fuse_reserve_datasets(geo, sat, how="left")
    assert len(result.data) == 10
    assert result.matched == 5
    assert result.unmatched_geological == 5
