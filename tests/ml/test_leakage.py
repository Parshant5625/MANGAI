"""Phase 2 tests: leakage protection, matrix compatibility, data-quality calculations."""

from __future__ import annotations

import pandas as pd

from ml.common.contracts import LEAKAGE_COLUMNS
from ml.common.validation import check_leakage
from ml.reserve.features import LEAKAGE_EXCLUSIONS, prepare_reserve_matrix
from ml.reserve.fusion import fuse_reserve_datasets

# ============================================================
# E. Leakage protection
# ============================================================


def test_leakage_registry_covers_all_reserve_targets():
    assert {"mn_pct", "fe_pct", "sio2_pct", "ore_thickness_m", "is_manganese"}.issubset(LEAKAGE_COLUMNS)


def test_leakage_registry_covers_all_production_targets():
    assert {"production_mt", "target_mt", "production_gap_mt", "shortfall"}.issubset(LEAKAGE_COLUMNS)


def test_reserve_matrix_excludes_every_registered_target(demo_store):
    fused = fuse_reserve_datasets(demo_store.geological(), demo_store.satellite(), validate=False).data
    matrix = prepare_reserve_matrix(fused)
    result = check_leakage(matrix, LEAKAGE_EXCLUSIONS, context="reserve")
    assert result.valid, result.errors


def test_check_leakage_detects_a_leaked_target():
    frame = pd.DataFrame({"feature_a": [1.0], "mn_pct": [12.0]})
    result = check_leakage(frame, ["mn_pct"], context="reserve")
    assert not result.valid
    assert any("mn_pct" in error for error in result.errors)


def test_production_feature_matrix_has_no_target_information(demo_store):
    from ml.production.features import FEATURE_COLUMNS, build_daily_features

    features = build_daily_features(demo_store.production())
    result = check_leakage(features[FEATURE_COLUMNS], ["production_mt", "target_mt"], context="production")
    assert result.valid, result.errors


def test_production_features_are_strictly_lagged(demo_store):
    from ml.production.features import build_daily_features

    production = demo_store.production().reset_index(drop=True)
    features = build_daily_features(production)
    last_index = features.index[-1]
    assert features.loc[last_index, "production_lag_1"] == production["production_mt"].iloc[-2]


def test_production_csv_no_longer_persists_the_shortfall_target(demo_store):
    """Regression guard: the derived shortfall target must not sit next to features on disk."""
    production = demo_store.production()
    assert "shortfall" not in production.columns


# ============================================================
# I. Numeric feature matrix compatibility
# ============================================================


def test_reserve_contract_columns_cover_the_reserve_feature_list(demo_store):
    fused = fuse_reserve_datasets(demo_store.geological(), demo_store.satellite(), validate=False).data
    for feature in ("elevation_m", "slope_deg", "aspect_deg", "depth_m", "ndvi", "swir_ratio"):
        assert feature in fused.columns, f"fusion lost reserve feature {feature}"


def test_fused_matrix_is_numeric_and_complete(demo_store):
    fused = fuse_reserve_datasets(demo_store.geological(), demo_store.satellite(), validate=False).data
    matrix = prepare_reserve_matrix(fused)
    assert len(matrix) == len(fused)
    assert not matrix.isna().any().any()
