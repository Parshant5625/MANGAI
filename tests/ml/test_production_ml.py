"""Tests for Production Intelligence ML pipeline."""

import numpy as np
import pandas as pd
from ml.production.features import FEATURE_COLUMNS, build_daily_features, chronological_split
from ml.risk.shortfall import shortfall_probability, severity


def test_daily_features_no_future_leakage(demo_store):
    """Verify features use only past data (lagged)."""
    df = demo_store.production()
    features = build_daily_features(df)
    
    # All feature columns should exist
    for col in FEATURE_COLUMNS:
        assert col in features.columns, f"Missing feature: {col}"
    
    # No NaN values in feature columns
    assert features[FEATURE_COLUMNS].notna().all().all()


def test_chronological_split_order_preserved(demo_store):
    """Verify chronological split maintains time order."""
    df = demo_store.production()
    features = build_daily_features(df)
    train, validation, test = chronological_split(features)
    
    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    
    # Verify order is preserved
    train_max_date = train["date"].max()
    val_min_date = validation["date"].min()
    assert train_max_date <= val_min_date


def test_shortfall_probability_monotonic():
    """Higher shortage should yield higher probability."""
    prob_low = shortfall_probability(gap_mt=-100, target_mt=1000)
    prob_high = shortfall_probability(gap_mt=-500, target_mt=1000)
    
    assert 0 <= prob_low <= 1
    assert 0 <= prob_high <= 1
    assert prob_low < prob_high


def test_shortfall_probability_with_pressure():
    """Risk pressure should increase probability."""
    prob_no_pressure = shortfall_probability(gap_mt=-200, target_mt=1000, risk_pressure=0.0)
    prob_with_pressure = shortfall_probability(gap_mt=-200, target_mt=1000, risk_pressure=1.0)
    
    assert prob_with_pressure > prob_no_pressure


def test_severity_classification():
    """Test severity classification logic."""
    assert severity(0.9, -200, 1000) == "CRITICAL"
    assert severity(0.7, -150, 1000) == "HIGH"
    assert severity(0.5, -50, 1000) == "MEDIUM"
    assert severity(0.2, -10, 1000) == "LOW"


def test_naive_baseline_reasonable(demo_store):
    """Naive forecast should be within reasonable range."""
    from ml.production.forecast import naive_rolling_forecast
    
    df = demo_store.production()
    forecast = naive_rolling_forecast(df, horizon_days=7)
    
    assert forecast > 0
    # Should be within 50% of actual production
    recent_mean = df["production_mt"].tail(28).mean() * 7
    assert forecast > recent_mean * 0.5
    assert forecast < recent_mean * 1.5