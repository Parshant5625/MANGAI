"""Deterministic spatial prediction grid for reserve intelligence.

For each grid cell the pipeline:

1. locates the nearest fused observation (lat/lon) and reuses its terrain +
   satellite covariates — this is the available geological/terrain/satellite
   context, NOT fabricated geology (documented as nearest-neighbour context);
2. constructs the exact training feature matrix for that cell;
3. scores prospectivity (baseline + calibrated ensemble when available),
   grade and thickness with conformal prediction intervals, and prototype
   resource potential;
4. attaches a data-support / extrapolation indicator.

Resolution is configurable and capped so the grid stays laptop-friendly.
If a cell has no context within the search radius it is returned with a
clear ``no_context`` data-support state rather than fabricated values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.reserve.features import (
    load_fused_reserve_table,
    project_root,
)
from ml.reserve.inference import (
    load_model_bundle,
    load_model_metadata,
    predict_with_model,
    resolve_model_path,
)
from ml.reserve.support import SupportAssessor

MAX_CELLS_PER_SIDE = 40
DEFAULT_CELLS_PER_SIDE = 10
SEARCH_RADIUS_M = 50_000.0
LAT_METRES = 110_574.0
LON_METRES = 103_000.0

# Columns reused as cell covariates from the nearest observation.
COVARIATE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "depth_m",
    "blue_b2",
    "green_b3",
    "red_b4",
    "nir_b8",
    "swir_b11",
    "swir_b12",
    "formation",
]


def _build_grid(bbox: dict[str, float], cells_per_side: int) -> pd.DataFrame:
    lats = np.linspace(bbox["min_lat"], bbox["max_lat"], cells_per_side)
    lons = np.linspace(bbox["min_lon"], bbox["max_lon"], cells_per_side)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return pd.DataFrame({"latitude": lat_grid.ravel(), "longitude": lon_grid.ravel()})


def _nearest_covariates(grid: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest-observation covariates to every grid cell."""
    obs = observations[["latitude", "longitude", *COVARIATE_COLUMNS]].copy()
    obs["_lat_m"] = obs["latitude"] * LAT_METRES
    obs["_lon_m"] = obs["longitude"] * LON_METRES
    grid = grid.copy()
    grid["_lat_m"] = grid["latitude"] * LAT_METRES
    grid["_lon_m"] = grid["longitude"] * LON_METRES
    nearest_idx = []
    distances_m = []
    # ``itertuples(..., name=None)`` yields plain tuples so leading-underscore
    # column names (``_lat_m``/``_lon_m``) stay positionally addressable across
    # pandas 2.x and 3.x (pandas 3 renamed them to ``_2``/``_3`` in namedtuples).
    for row in grid.itertuples(index=False, name=None):
        lat_m, lon_m = row[2], row[3]
        d2 = (obs["_lat_m"] - lat_m) ** 2 + (obs["_lon_m"] - lon_m) ** 2
        idx = int(d2.idxmin())
        nearest_idx.append(idx)
        distances_m.append(float(d2.iloc[idx] ** 0.5))
    context = obs.iloc[nearest_idx].reset_index(drop=True)
    for column in COVARIATE_COLUMNS:
        grid[column] = context[column].to_numpy()
    grid["nearest_observation_m"] = distances_m
    grid["context_available"] = [d <= SEARCH_RADIUS_M for d in distances_m]
    return grid


def generate_prediction_grid(
    model_dir: Path,
    *,
    bbox: dict[str, float] | None = None,
    cells_per_side: int = DEFAULT_CELLS_PER_SIDE,
    observations: pd.DataFrame | None = None,
    coverage: float = 0.9,
) -> dict[str, Any]:
    """Build a scored spatial prediction grid.

    ``observations`` is the fused geological+satellite table used for
    nearest-neighbour context; if omitted it is loaded from the synthetic
    dataset. ``model_dir`` is the directory containing trained artifacts.
    """
    model_dir = Path(model_dir)
    root = project_root()
    if observations is None:
        observations = load_fused_reserve_table(root)
    if bbox is None:
        pad = 0.01
        bbox = {
            "min_lat": float(observations["latitude"].min()) - pad,
            "max_lat": float(observations["latitude"].max()) + pad,
            "min_lon": float(observations["longitude"].min()) - pad,
            "max_lon": float(observations["longitude"].max()) + pad,
        }
    cells_per_side = int(min(max(cells_per_side, 2), MAX_CELLS_PER_SIDE))

    grid = _build_grid(bbox, cells_per_side)
    grid = _nearest_covariates(grid, observations)

    prospectivity_path = resolve_model_path(model_dir, "prospectivity")
    ensemble_path = resolve_model_path(model_dir, "prospectivity_ensemble")
    grade_path = resolve_model_path(model_dir, "grade")
    thickness_path = resolve_model_path(model_dir, "thickness")

    support = _fit_support(model_dir, ensemble_path, observations)

    cells: list[dict[str, Any]] = []
    for row in grid.itertuples(index=False):
        cells.append(
            _score_cell(
                row,
                model_dir=model_dir,
                prospectivity_path=prospectivity_path,
                ensemble_path=ensemble_path,
                grade_path=grade_path,
                thickness_path=thickness_path,
                coverage=coverage,
                support=support,
            )
        )

    return {
        "bbox": bbox,
        "cells_per_side": cells_per_side,
        "cells": cells,
        "model_versions": {
            "prospectivity": _version_of(prospectivity_path),
            "prospectivity_ensemble": _version_of(ensemble_path),
            "grade": _version_of(grade_path),
            "thickness": _version_of(thickness_path),
        },
        "context_method": "nearest_observation_covariates",
        "search_radius_m": SEARCH_RADIUS_M,
        "synthetic_data": True,
    }


def _version_of(model_path) -> str | None:
    if model_path is None:
        return None
    meta = load_model_metadata(model_path)
    return meta.get("version") if meta else None


def _fit_support(model_dir, ensemble_path, observations: pd.DataFrame) -> SupportAssessor | None:
    """Reuse ensemble support stats when available; otherwise fit on covariates."""
    if ensemble_path is not None:
        meta = load_model_metadata(ensemble_path)
        if meta and "support_stats" in meta:
            try:
                return SupportAssessor.from_dict(meta["support_stats"])
            except Exception:
                pass
    numeric = observations[[c for c in COVARIATE_COLUMNS if c in observations.columns and c != "formation"]]
    if numeric.empty:
        return None
    return SupportAssessor.fit(numeric, observations["latitude"], observations["longitude"])


def _score_cell(
    row,
    *,
    model_dir: Path,
    prospectivity_path,
    ensemble_path,
    grade_path,
    thickness_path,
    coverage: float,
    support: SupportAssessor | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "latitude": round(float(row.latitude), 6),
        "longitude": round(float(row.longitude), 6),
        "context_available": bool(row.context_available),
        "nearest_observation_m": round(float(row.nearest_observation_m), 1),
    }
    if not row.context_available:
        # Withheld prediction: the full cell contract is still emitted so API
        # consumers can rely on every key, but every model output is ``None``.
        base.update(
            {
                "probability": None,
                "calibrated_probability": None,
                "base_probabilities": None,
                "predicted_grade_pct": None,
                "grade_interval": None,
                "predicted_thickness_m": None,
                "thickness_interval": None,
                "resource_potential": None,
                "extrapolation_level": None,
                "model_version": None,
                "data_support": {
                    "state": "no_context",
                    "note": "No observation within search radius; prediction withheld rather than fabricated.",
                },
            }
        )
        return base

    cell_frame = pd.DataFrame(
        [
            {
                "latitude": row.latitude,
                "longitude": row.longitude,
                "elevation_m": row.elevation_m,
                "slope_deg": row.slope_deg,
                "aspect_deg": row.aspect_deg,
                "depth_m": row.depth_m,
                "blue_b2": row.blue_b2,
                "green_b3": row.green_b3,
                "red_b4": row.red_b4,
                "nir_b8": row.nir_b8,
                "swir_b11": row.swir_b11,
                "swir_b12": row.swir_b12,
                "formation": row.formation,
            }
        ]
    )
    features = {
        name: float(cell_frame[name].iloc[0])
        for name in cell_frame.columns
        if pd.api.types.is_numeric_dtype(cell_frame[name])
    }

    probability: float | None = None
    calibrated_probability: float | None = None
    base_probabilities: dict[str, float] | None = None
    model_version: str | None = None

    if ensemble_path is not None:
        try:
            ensemble, columns = load_model_bundle(ensemble_path, task="classification")
            from ml.reserve.ensemble import ReserveEnsemble

            if isinstance(ensemble, ReserveEnsemble):
                aligned = cell_frame.reindex(columns=columns, fill_value=0)
                raw = float(ensemble.raw_probabilities(aligned)[0])
                calibrated_probability = float(ensemble.calibrator.transform(np.array([raw]))[0])
                base_probabilities = ensemble.base_probabilities(aligned)
                probability = calibrated_probability
                model_version = _version_of(ensemble_path)
        except Exception:
            ensemble_path = None

    if probability is None and prospectivity_path is not None:
        try:
            probs = predict_with_model(cell_frame, prospectivity_path, "classification")
            probability = float(probs.iloc[0])
            model_version = _version_of(prospectivity_path)
        except Exception:
            probability = None

    grade, grade_interval = _regressor_interval(cell_frame, grade_path, model_dir, coverage)
    thickness, thickness_interval = _regressor_interval(cell_frame, thickness_path, model_dir, coverage, non_negative=True)

    resource_potential = None
    if probability is not None and thickness is not None:
        low = thickness_interval.get("lower") if thickness_interval else thickness
        high = thickness_interval.get("upper") if thickness_interval else thickness
        resource_potential = _resource_potential(probability, thickness, low, high, model_version)

    data_support: dict[str, Any] = {"state": "context_available"}
    extrapolation_level = None
    if support is not None:
        assessment = support.assess(features, latitude=row.latitude, longitude=row.longitude)
        data_support.update(assessment.get("data_support", {}))
        extrapolation_level = assessment.get("extrapolation", {}).get("level")

    return {
        **base,
        "probability": None if probability is None else round(probability, 4),
        "calibrated_probability": None if calibrated_probability is None else round(calibrated_probability, 4),
        "base_probabilities": base_probabilities,
        "predicted_grade_pct": None if grade is None else round(grade, 2),
        "grade_interval": grade_interval,
        "predicted_thickness_m": None if thickness is None else round(thickness, 2),
        "thickness_interval": thickness_interval,
        "resource_potential": resource_potential,
        "data_support": data_support,
        "extrapolation_level": extrapolation_level,
        "model_version": model_version,
    }


def _regressor_interval(cell_frame, model_path, model_dir, coverage: float, *, non_negative: bool = False):
    if model_path is None:
        return None, None
    try:
        prediction = float(predict_with_model(cell_frame, model_path, "regression").iloc[0])
    except Exception:
        return None, None
    interval = _conformal_interval(model_path, model_dir, prediction, coverage, non_negative=non_negative)
    return prediction, interval


def _conformal_interval(model_path, model_dir, prediction: float, coverage: float, *, non_negative: bool) -> dict[str, Any] | None:
    from ml.reserve.conformal import SplitConformalRegressor

    state = _load_conformal_state(model_path, model_dir)
    if state is None:
        return None
    calibrator = SplitConformalRegressor.from_dict(state)
    interval = calibrator.interval(prediction)
    interval["coverage"] = coverage
    interval["non_negative"] = non_negative
    return interval


def _load_conformal_state(model_path, model_dir) -> dict[str, Any] | None:
    stem = model_path.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    for candidate in (
        model_dir / "reserve" / f"{stem}_conformal.json",
        model_path.with_name(f"{stem}_conformal.json"),
    ):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _resource_potential(
    probability: float,
    thickness: float,
    thickness_low: float,
    thickness_high: float,
    model_version: str | None,
) -> dict[str, Any]:
    from ml.reserve.resource_estimator import estimate_resource_potential_with_intervals

    estimate = estimate_resource_potential_with_intervals(
        probability=probability,
        thickness_point=thickness,
        thickness_low=thickness_low,
        thickness_high=thickness_high,
    )
    return {
        "label": "prototype resource potential",
        "expected_tonnage": round(estimate.expected_tonnage, 2),
        "p10": round(estimate.p10, 2),
        "p50": round(estimate.p50, 2),
        "p90": round(estimate.p90, 2),
        "assumptions": estimate.assumptions,
        "uncertainty_sources": estimate.uncertainty_sources,
        "model_version": model_version,
    }
