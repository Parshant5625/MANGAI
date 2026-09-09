"""Deterministic fusion of geological + satellite observations for reserve modeling.

This module owns the single canonical alignment between geological sample
observations and satellite features. Both datasets remain conceptually
separate; this layer performs the spatial/context alignment that produces
the reserve feature dataset consumed by feature engineering and ML.

Architecture:

    geological observations
            +
    satellite observations
            +
    terrain/context
            ↓
    spatial/context alignment (this module)
            ↓
    reserve feature dataset
            ↓
    feature engineering
            ↓
    ML
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.common.contracts import GEOLOGICAL_CONTRACT, SATELLITE_CONTRACT
from ml.common.validation import validate_dataset

# Columns used to align geological and satellite observations.
ALIGNMENT_KEYS = ("sample_id", "latitude", "longitude")


def _alignment_columns() -> list[str]:
    """Return alignment keys as a list (pandas column selection requires a list)."""
    return list(ALIGNMENT_KEYS)


@dataclass
class FusionResult:
    """Outcome of a geological + satellite fusion."""

    data: pd.DataFrame
    matched: int
    unmatched_geological: int
    unmatched_satellite: int

    @property
    def merge_rate(self) -> float:
        total = self.matched + self.unmatched_geological
        if total == 0:
            return 0.0
        return self.matched / total


def fuse_reserve_datasets(
    geological: pd.DataFrame,
    satellite: pd.DataFrame,
    *,
    how: str = "inner",
    validate: bool = True,
) -> FusionResult:
    """Fuse geological and satellite observations on alignment keys.

    Args:
        geological: DataFrame matching ``GEOLOGICAL_CONTRACT``.
        satellite: DataFrame matching ``SATELLITE_CONTRACT``.
        how: Join strategy. ``"inner"`` keeps only matched samples.
        validate: When True, validate inputs against their contracts first.

    Returns:
        FusionResult with fused data and match statistics.

    Raises:
        ValueError: When validation fails and ``validate=True``.
    """
    if validate:
        geo_result = validate_dataset(geological, GEOLOGICAL_CONTRACT)
        geo_result.raise_for_errors()
        sat_result = validate_dataset(satellite, SATELLITE_CONTRACT)
        sat_result.raise_for_errors()

    keys = _alignment_columns()
    merged = geological.merge(satellite, on=keys, how=how, suffixes=("", "_sat"))

    # A satellite-owned column (required by contract) doubles as the join-match
    # sentinel: rows where it is null never carried satellite data.
    sentinel = "nir_b8"
    if sentinel in merged.columns:
        matched = int(merged[sentinel].notna().sum())
    else:
        matched = len(merged)

    geo_merge_check = geological[keys].merge(satellite[keys], on=keys, how="left", indicator=True)
    unmatched_geological = int((geo_merge_check["_merge"] == "left_only").sum())
    sat_merge_check = satellite[keys].merge(geological[keys], on=keys, how="left", indicator=True)
    unmatched_satellite = int((sat_merge_check["_merge"] == "left_only").sum())

    return FusionResult(
        data=merged,
        matched=matched,
        unmatched_geological=unmatched_geological,
        unmatched_satellite=unmatched_satellite,
    )


def load_fused_reserve_table(
    geological: pd.DataFrame,
    satellite: pd.DataFrame,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper returning only the fused DataFrame."""
    return fuse_reserve_datasets(geological, satellite, validate=validate).data
