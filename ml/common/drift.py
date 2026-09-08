"""Lightweight drift-monitoring foundation for MANGAI reserve models.

This is a FOUNDATION for later MLOps — it does NOT automatically declare a
model invalid. It compares training feature distributions against new
inference feature distributions and returns per-feature metrics plus a
``warning``/``ok`` status.

Supported metrics (all documented, all simple):
- missingness drift: absolute change in the null fraction
- mean / median shift: standardized by the training standard deviation
- percentile shift: standardized shift at the configured percentiles
- PSI (Population Stability Index): 10-quantile binning on the training
  distribution, applied to the current distribution
- categorical distribution shift: total-variation distance on category
  frequencies (when categories are provided)

Thresholds are configurable and documented; exceeding a threshold yields a
``warning`` status, never an automatic invalidation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PERCENTILES = (10, 50, 90)
DEFAULT_PSI_BINS = 10
DEFAULT_THRESHOLDS = {
    "missingness": 0.05,
    "mean_shift": 0.25,
    "median_shift": 0.25,
    "percentile_shift": 0.25,
    "psi": 0.25,
    "categorical_tv": 0.15,
}


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = DEFAULT_PSI_BINS) -> float:
    """Population Stability Index between two 1-D arrays."""
    finite_ref = reference[np.isfinite(reference)]
    if len(finite_ref) < bins:
        return 0.0
    edges = np.quantile(finite_ref, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * len(ref_counts))
    cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * len(cur_counts))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _total_variation(reference: pd.Series, current: pd.Series) -> float:
    categories = set(reference.unique()) | set(current.unique())
    ref_frac = reference.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_frac = current.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return float(0.5 * np.abs(ref_frac - cur_frac).sum())


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
    psi_bins: int = DEFAULT_PSI_BINS,
    thresholds: dict[str, float] | None = None,
    categorical_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Compare training (``reference``) vs new inference (``current``) features.

    Returns per-feature metrics and an overall status. A ``warning`` status
    means at least one metric exceeded its configurable threshold; it never
    invalidates the model.
    """
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    categorical_columns = categorical_columns or []
    common = [column for column in reference.columns if column in current.columns]
    features: dict[str, Any] = {}
    warnings: list[str] = []

    for column in common:
        ref = reference[column]
        cur = current[column]
        if column in categorical_columns or ref.dtype == object or cur.dtype == object:
            tv = _total_variation(ref.astype(str), cur.astype(str))
            status = "warning" if tv > thresholds["categorical_tv"] else "ok"
            features[column] = {
                "kind": "categorical",
                "total_variation": round(tv, 4),
                "threshold": thresholds["categorical_tv"],
                "status": status,
            }
            if status == "warning":
                warnings.append(column)
            continue

        ref_arr = pd.to_numeric(ref, errors="coerce").to_numpy()
        cur_arr = pd.to_numeric(cur, errors="coerce").to_numpy()
        ref_finite = ref_arr[np.isfinite(ref_arr)]
        cur_finite = cur_arr[np.isfinite(cur_arr)]
        if len(ref_finite) == 0 or len(cur_finite) == 0:
            features[column] = {"kind": "numeric", "status": "ok", "note": "insufficient data"}
            continue

        ref_missing = float(np.mean(~np.isfinite(ref_arr)))
        cur_missing = float(np.mean(~np.isfinite(cur_arr)))
        missing_delta = abs(cur_missing - ref_missing)
        std = float(np.std(ref_finite)) or 1.0
        mean_shift = abs(float(np.mean(cur_finite)) - float(np.mean(ref_finite))) / std
        median_shift = abs(float(np.median(cur_finite)) - float(np.median(ref_finite))) / std
        percentile_shift = 0.0
        for percentile in percentiles:
            ref_p = float(np.percentile(ref_finite, percentile))
            cur_p = float(np.percentile(cur_finite, percentile))
            percentile_shift = max(percentile_shift, abs(cur_p - ref_p) / std)
        psi = _psi(ref_finite, cur_finite, bins=psi_bins)

        metric_values = {
            "missingness": missing_delta,
            "mean_shift": mean_shift,
            "median_shift": median_shift,
            "percentile_shift": percentile_shift,
            "psi": psi,
        }
        status = "ok"
        for metric, value in metric_values.items():
            if value > thresholds.get(metric, float("inf")):
                status = "warning"
                warnings.append(column)
                break
        features[column] = {
            "kind": "numeric",
            "metrics": {name: round(value, 4) for name, value in metric_values.items()},
            "thresholds": {name: thresholds.get(name, float("inf")) for name in metric_values},
            "status": status,
        }

    return {
        "features": features,
        "features_with_warnings": sorted(set(warnings)),
        "overall_status": "warning" if warnings else "ok",
        "config": {"percentiles": list(percentiles), "psi_bins": psi_bins, "thresholds": thresholds},
        "note": "warning status does not invalidate the model; it is a monitoring signal only.",
    }
