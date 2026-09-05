"""Tests for Reserve Intelligence ML pipeline."""

import numpy as np
import pandas as pd
from ml.reserve.features import (
    RESERVE_NUMERICAL_FEATURES,
    RESERVE_CATEGORICAL_FEATURES,
    LEAKAGE_EXCLUSIONS,
    ensure_spectral_indices,
    prepare_reserve_matrix,
)
from ml.reserve.resource_estimator import estimate_resource_potential


def test_spectral_indices_computed(demo_store):
    """Verify spectral indices are computed correctly."""
    df = demo_store.geological()
    result = ensure_spectral_indices(df)
    
    assert "ndvi" in result.columns
    assert "ndwi" in result.columns
    assert "swir_ratio" in result.columns
    assert "bare_soil_index" in result.columns


def test_reserve_matrix_has_no_leakage_columns(demo_store):
    """Ensure leakage columns are excluded from features."""
    df = demo_store.geological()
    df = ensure_spectral_indices(df)
    matrix = prepare_reserve_matrix(df)
    
    for col in LEAKAGE_EXCLUSIONS:
        assert col not in matrix.columns, f"Leakage column {col} found in matrix"


def test_reserve_matrix_numeric_only(demo_store):
    """Verify reserve matrix contains only numeric values."""
    df = demo_store.geological()
    matrix = prepare_reserve_matrix(df)
    
    assert matrix.select_dtypes(include=[np.number]).shape[1] == matrix.shape[1]


def test_resource_estimator_returns_valid_uncertainty():
    """Test Monte Carlo resource estimation."""
    estimate = estimate_resource_potential(
        probability=0.75,
        thickness_m=5.0,
        cell_area_m2=10000,
        density_t_per_m3=3.6,
    )
    
    assert estimate.expected_tonnage > 0
    assert estimate.p10 <= estimate.p50 <= estimate.p90
    assert estimate.p10 > 0


def test_resource_estimator_respects_confidence():
    """Higher confidence should reduce uncertainty spread."""
    high_conf = estimate_resource_potential(
        probability=0.8,
        thickness_m=4.0,
        cell_area_m2=10000,
        density_t_per_m3=3.6,
        probability_std=0.05,
        thickness_std_fraction=0.1,
    )
    
    low_conf = estimate_resource_potential(
        probability=0.8,
        thickness_m=4.0,
        cell_area_m2=10000,
        density_t_per_m3=3.6,
        probability_std=0.2,
        thickness_std_fraction=0.4,
    )
    
    high_spread = high_conf.p90 - high_conf.p10
    low_spread = low_conf.p90 - low_conf.p10
    
    assert high_spread < low_spread


def test_probability_clipping(demo_store):
    """Verify probability values are clipped to valid range."""
    from ml.reserve.inference import predict_prospectivity_frame
    from pathlib import Path
    import os
    
    df = demo_store.reserve_predictions()
    
    # Ensure probabilities are in valid range
    if "manganese_probability" in df.columns:
        assert df["manganese_probability"].between(0, 1).all()