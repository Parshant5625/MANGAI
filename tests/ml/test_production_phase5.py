from __future__ import annotations

import pandas as pd
import pytest

from ml.production.features import FEATURE_COLUMNS, build_daily_features
from ml.production.pipeline import FORBIDDEN_FEATURES, add_future_targets, assert_no_target_leakage, chronological_splits, load_fused_production_data


def test_production_feature_columns_exclude_forbidden_targets() -> None:
    assert not set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES
    assert_no_target_leakage(FEATURE_COLUMNS)


def test_fused_data_is_one_row_per_date() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    fused = load_fused_production_data(root)
    assert fused["date"].is_unique
    assert len(fused) >= 365


def test_future_targets_shift_forward_without_leakage() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40),
            "production_mt": range(40),
            "target_mt": [10] * 40,
            "rainfall_mm": [1.0] * 40,
        }
    )
    target = add_future_targets(df, 7)
    assert target.loc[0, "future_production_mt"] == 7
    assert target.loc[0, "shortfall_label"] is True or target.loc[0, "shortfall_label"] == 1
    features = build_daily_features(target)
    assert not set(FEATURE_COLUMNS) & {"future_production_mt", "shortfall_label"}
    train, validation, test = chronological_splits(features)
    assert train["date"].max() < validation["date"].min() <= test["date"].min()


def test_chronological_split_rejects_tiny_input() -> None:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2)})
    with pytest.raises(ValueError):
        chronological_splits(df)
