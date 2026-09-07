from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import get_settings
from backend.app.core.errors import ModelUnavailableError
from backend.app.services.demo_data import DemoDataStore, demo_envelope, heuristic_prospectivity
from backend.app.services.model_artifacts import reserve_prospectivity_available
from ml.reserve.explain import explain_row
from ml.reserve.features import ensure_spectral_indices
from ml.reserve.grid import generate_prediction_grid
from ml.reserve.inference import (
    load_model_metadata,
    load_support_assessor,
    predict_ensemble_frame,
    predict_prospectivity_frame,
    predict_regressor_with_interval,
    resolve_ensemble_path,
    resolve_model_path,
)
from ml.reserve.resource_estimator import estimate_resource_potential_with_intervals

CELL_AREA_M2 = 10_000.0
DENSITY_T_PER_M3 = 3.6
_FRAME_CACHE: dict[tuple[str, float], pd.DataFrame] = {}
logger = logging.getLogger(__name__)


class ReserveService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def _model_dir(self):
        return self.settings.resolved_model_dir

    def _cache_key(self) -> tuple[str, float]:
        model_dir = self._model_dir()
        stamps = []
        for path in [
            model_dir / "reserve" / "prospectivity_model.joblib",
            model_dir / "reserve" / "prospectivity_features.pkl",
            model_dir / "reserve" / "prospectivity_xgboost.json",
            model_dir / "reserve" / "prospectivity_xgboost_features.pkl",
            model_dir / "reserve_xgboost.json",
            model_dir / "reserve_features.pkl",
            model_dir / "reserve" / "grade_model.joblib",
            model_dir / "reserve" / "grade_xgboost.json",
            model_dir / "reserve" / "grade_xgboost_features.pkl",
            model_dir / "reserve" / "thickness_model.joblib",
            model_dir / "reserve" / "thickness_xgboost.json",
            model_dir / "reserve" / "thickness_xgboost_features.pkl",
            self.settings.resolved_data_dir / "processed" / "reserve_predictions.csv",
        ]:
            stamps.append(path.stat().st_mtime if path.exists() else 0.0)
        return (str(model_dir), max(stamps) if stamps else 0.0)

    def _base_frame(self) -> pd.DataFrame:
        key = self._cache_key()
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            return cached.copy()
        frame = self._compute_base_frame()
        _FRAME_CACHE.clear()
        _FRAME_CACHE[key] = frame
        return frame.copy()

    def _compute_base_frame(self) -> pd.DataFrame:
        df = ensure_spectral_indices(self.store.reserve_predictions().copy())
        model_dir = self._model_dir()
        ensemble_path = resolve_ensemble_path(model_dir)
        try:
            if ensemble_path is not None:
                scored = predict_ensemble_frame(df, model_dir)
                if scored is not None:
                    df["manganese_probability"] = scored["manganese_probability"]
                    df["prospectivity_class"] = scored["prospectivity_class"]
                else:
                    raise FileNotFoundError("ensemble scoring returned None")
            else:
                scored = predict_prospectivity_frame(df, model_dir)
                df["manganese_probability"] = scored["manganese_probability"]
                df["prospectivity_class"] = scored["prospectivity_class"]
        except Exception as exc:
            logger.warning("Reserve prospectivity model unavailable; using demo heuristic fallback: %s", exc)
            if self.settings.require_model_artifacts:
                raise ModelUnavailableError(
                    "Reserve prospectivity model is not available.",
                    details={"model": "reserve_prospectivity"},
                ) from exc
            df["manganese_probability"] = heuristic_prospectivity(df)
            df["prospectivity_class"] = pd.cut(
                df["manganese_probability"],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
                include_lowest=True,
            ).astype(str)
        df["manganese_probability"] = df["manganese_probability"].astype(float).clip(0.01, 0.99)

        grade, grade_fn = predict_regressor_with_interval(df, model_dir, "grade")
        thickness, thickness_fn = predict_regressor_with_interval(df, model_dir, "thickness")
        df["predicted_grade_pct"] = grade.clip(2, 48).round(2) if grade is not None else self._predict_grade(df)
        df["predicted_thickness_m"] = thickness.clip(0.2, 18).round(2) if thickness is not None else self._predict_thickness(df)
        # Store conformal interval bounds as columns for downstream consumers.
        df["grade_interval_lower"] = df["grade_interval_upper"] = None
        df["thickness_interval_lower"] = df["thickness_interval_upper"] = None
        if grade is not None and grade_fn is not None:
            intervals = grade_fn(grade)
            df["grade_interval_lower"] = round(intervals["lower"], 2)
            df["grade_interval_upper"] = round(intervals["upper"], 2)
        if thickness is not None and thickness_fn is not None:
            intervals = thickness_fn(thickness)
            df["thickness_interval_lower"] = round(intervals["lower"], 2)
            df["thickness_interval_upper"] = round(intervals["upper"], 2)
        df["confidence"] = self._confidence(df)
        return df

    def _predict_grade(self, df: pd.DataFrame) -> pd.Series:
        formation = df["formation"].astype(str).str.contains("Manganiferous|Gondite", case=False, regex=True)
        swir = (df["swir_ratio"] - df["swir_ratio"].min()) / (df["swir_ratio"].max() - df["swir_ratio"].min() + 1e-6)
        ndvi = (df["ndvi"] - df["ndvi"].min()) / (df["ndvi"].max() - df["ndvi"].min() + 1e-6)
        grade = 7 + df["manganese_probability"].astype(float) * 26 + swir * 5 + formation.astype(float) * 4 - ndvi * 2
        return grade.clip(2, 48).round(2)

    def _predict_thickness(self, df: pd.DataFrame) -> pd.Series:
        probability = df["manganese_probability"].astype(float)
        depth = df["depth_m"].astype(float)
        depth_window = ((depth >= 10) & (depth <= 55)).astype(float)
        slope_penalty = ((df["slope_deg"] - df["slope_deg"].min()) / (df["slope_deg"].max() - df["slope_deg"].min() + 1e-6)) * 2.2
        formation_bonus = df["formation"].astype(str).str.contains("Manganiferous|Gondite", case=False, regex=True)
        thickness = 0.8 + probability * 10.5 + depth_window * 2.0 + formation_bonus.astype(float) * 1.4
        return (thickness - slope_penalty).clip(0.2, 18).round(2)

    def _confidence(self, df: pd.DataFrame) -> pd.Series:
        probability = df["manganese_probability"].astype(float)
        support = np.minimum(1.0, 0.55 + np.abs(probability - 0.5) * 0.8)
        spectral_completeness = 1 - df[["blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12"]].isna().mean(axis=1)
        return pd.Series((support * 0.7 + spectral_completeness * 0.3).clip(0.35, 0.95), index=df.index).round(2)

    def _parse_bbox(self, bbox: str) -> list[float]:
        try:
            values = [float(part.strip()) for part in bbox.split(",")]
        except ValueError as exc:
            raise ValueError("bbox must contain numeric min_lon,min_lat,max_lon,max_lat") from exc
        if len(values) != 4:
            raise ValueError("bbox must contain min_lon,min_lat,max_lon,max_lat")
        min_lon, min_lat, max_lon, max_lat = values
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("bbox longitude values must be between -180 and 180")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("bbox latitude values must be between -90 and 90")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox minimum coordinates must be smaller than maximum coordinates")
        return values

    def _resource_payload(
        self,
        probability: float,
        thickness_m: float,
        thickness_low: float | None,
        thickness_high: float | None,
        seed_key: str,
    ) -> dict[str, Any]:
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        low = thickness_low if thickness_low is not None else thickness_m
        high = thickness_high if thickness_high is not None else thickness_m
        estimate = estimate_resource_potential_with_intervals(
            probability=probability,
            thickness_point=thickness_m,
            thickness_low=low,
            thickness_high=high,
            cell_area_m2=CELL_AREA_M2,
            density_t_per_m3=DENSITY_T_PER_M3,
            seed=seed,
        )
        return {
            "label": "prototype resource potential",
            "expected_tonnage": round(estimate.expected_tonnage, 2),
            "p10": round(estimate.p10, 2),
            "p50": round(estimate.p50, 2),
            "p90": round(estimate.p90, 2),
            "assumptions": estimate.assumptions,
            "uncertainty_sources": estimate.uncertainty_sources,
        }

    def _contributors(self, row: pd.Series) -> list[dict[str, Any]]:
        return [
            {"feature": "swir_ratio", "direction": "positive", "importance": 0.31, "value": round(float(row.get("swir_ratio", 0)), 3)},
            {"feature": "formation", "direction": "positive" if "Manganiferous" in str(row["formation"]) or "Gondite" in str(row["formation"]) else "neutral", "importance": 0.24, "value": str(row["formation"])},
            {"feature": "depth_m", "direction": "positive" if 10 <= float(row["depth_m"]) <= 55 else "neutral", "importance": 0.18, "value": round(float(row["depth_m"]), 2)},
        ]

    def _shap_contributors(self, row: pd.Series) -> list[dict[str, Any]]:
        model_path = resolve_model_path(self._model_dir(), "prospectivity")
        if model_path is None:
            return self._contributors(row)
        return explain_row(pd.DataFrame([row]), model_path) or self._contributors(row)

    def _model_metadata(self, name: str = "prospectivity") -> dict[str, Any] | None:
        """Training metadata for the currently served model artifact."""
        model_path = resolve_model_path(self._model_dir(), name)
        if model_path is None:
            return None
        return load_model_metadata(model_path)

    def _cell(self, row: pd.Series) -> dict[str, Any]:
        grade_low = float(row["grade_interval_lower"]) if "grade_interval_lower" in row.index and pd.notna(row.get("grade_interval_lower")) else None
        grade_high = float(row["grade_interval_upper"]) if "grade_interval_upper" in row.index and pd.notna(row.get("grade_interval_upper")) else None
        thickness_low = float(row["thickness_interval_lower"]) if "thickness_interval_lower" in row.index and pd.notna(row.get("thickness_interval_lower")) else None
        thickness_high = float(row["thickness_interval_upper"]) if "thickness_interval_upper" in row.index and pd.notna(row.get("thickness_interval_upper")) else None
        resource = self._resource_payload(
            float(row["manganese_probability"]),
            float(row["predicted_thickness_m"]),
            thickness_low,
            thickness_high,
            str(row["sample_id"]),
        )
        data_support: dict[str, Any] = {
            "spectral_bands_present": int(
                pd.Series(row[["blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12"]]).notna().sum()
            )
            if all(column in row.index for column in ["blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12"])
            else 0,
            "formation": str(row.get("formation", "")),
        }
        cell: dict[str, Any] = {
            "id": str(row["sample_id"]),
            "latitude": round(float(row["latitude"]), 6),
            "longitude": round(float(row["longitude"]), 6),
            "probability": round(float(row["manganese_probability"]), 4),
            "prospectivity_class": str(row["prospectivity_class"]),
            "predicted_grade_pct": round(float(row["predicted_grade_pct"]), 2),
            "predicted_thickness_m": round(float(row["predicted_thickness_m"]), 2),
            "confidence": round(float(row["confidence"]), 2),
            "resource_potential": resource,
            "top_contributors": self._contributors(row),
            "data_support": data_support,
        }
        # Phase 4 enrichment (optional, only when artifacts produced them).
        if "manganese_probability_raw" in row.index and pd.notna(row.get("manganese_probability_raw")):
            cell["calibrated_probability"] = round(float(row["manganese_probability"]), 4)
            cell["base_probabilities"] = self._row_base_probabilities(row)
        if grade_low is not None and grade_high is not None:
            cell["grade_interval"] = {
                "lower": round(grade_low, 2),
                "upper": round(grade_high, 2),
                "level": 0.9,
                "method": "split_conformal",
                "coverage": 0.9,
                "non_negative": False,
            }
        if thickness_low is not None and thickness_high is not None:
            cell["thickness_interval"] = {
                "lower": round(thickness_low, 2),
                "upper": round(thickness_high, 2),
                "level": 0.9,
                "method": "split_conformal",
                "coverage": 0.9,
                "non_negative": True,
            }
        support = load_support_assessor(self._model_dir())
        if support is not None:
            assessment = support.assess(
                {name: float(row.get(name, float("nan"))) for name in [
                    "elevation_m", "slope_deg", "depth_m", "ndvi", "ndwi", "swir_ratio",
                ] if name in row.index},
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )
            cell["extrapolation"] = assessment.get("extrapolation")
            cell["data_support_detail"] = assessment.get("data_support")
        return cell

    def _row_base_probabilities(self, row: pd.Series) -> dict[str, float]:
        ensemble_path = resolve_ensemble_path(self._model_dir())
        if ensemble_path is None:
            return {}
        try:
            from ml.reserve.inference import load_ensemble
            ensemble, columns = load_ensemble(self._model_dir())
            if ensemble is None:
                return {}
            from ml.reserve.features import ensure_spectral_indices, prepare_reserve_matrix
            frame = ensure_spectral_indices(pd.DataFrame([row.to_dict()]))
            return ensemble.base_probabilities(prepare_reserve_matrix(frame, columns))
        except Exception:
            return {}

    def get_prospectivity(
        self,
        site_id: str | None = None,
        bbox: str | None = None,
        min_probability: float | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        bbox_values: list[float] | None = None
        if bbox:
            bbox_values = self._parse_bbox(bbox)
        df = self._base_frame()
        if bbox_values:
            min_lon, min_lat, max_lon, max_lat = bbox_values
            df = df[
                (df["longitude"] >= min_lon)
                & (df["longitude"] <= max_lon)
                & (df["latitude"] >= min_lat)
                & (df["latitude"] <= max_lat)
            ]
        if min_probability is not None:
            df = df[df["manganese_probability"] >= min_probability]
        df = df.sort_values("manganese_probability", ascending=False).head(max(1, min(limit, 2000)))
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "bbox": bbox_values,
            "count": int(len(df)),
            "cells": [self._cell(row) for _, row in df.iterrows()],
        }

    def get_summary(self, site_id: str | None = None) -> dict[str, Any]:
        df = self._base_frame()
        high = df[df["manganese_probability"] >= 0.75]
        very_high = df[df["manganese_probability"] >= 0.85]
        payloads = [
            self._resource_payload(
                float(row["manganese_probability"]),
                float(row["predicted_thickness_m"]),
                float(row["confidence"]),
                str(row["sample_id"]),
            )
            for _, row in high.iterrows()
        ]
        total = {
            "label": "prototype resource potential",
            "expected_tonnage": round(float(sum(item["expected_tonnage"] for item in payloads)), 2),
            "p10": round(float(sum(item["p10"] for item in payloads)), 2),
            "p50": round(float(sum(item["p50"] for item in payloads)), 2),
            "p90": round(float(sum(item["p90"] for item in payloads)), 2),
            "assumptions": payloads[0]["assumptions"] if payloads else {
                "cell_area_m2": CELL_AREA_M2,
                "estimated_density_t_per_m3": DENSITY_T_PER_M3,
                "uncertainty_method": "Monte Carlo",
                "classification_boundary": "prototype only, not official reserves",
            },
        }
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "cells_evaluated": int(len(df)),
            "high_prospectivity_cells": int(len(high)),
            "very_high_prospectivity_cells": int(len(very_high)),
            "high_prospectivity_area_ha": round(len(high) * CELL_AREA_M2 / 10_000, 2),
            "average_probability": round(float(df["manganese_probability"].mean()), 4),
            "average_predicted_grade_pct": round(float(df["predicted_grade_pct"].mean()), 2),
            "average_predicted_thickness_m": round(float(df["predicted_thickness_m"].mean()), 2),
            "prototype_resource_potential": total,
            "validation_note": "Synthetic-data prototype with spatial-block validation. Not an official mineral resource or reserve.",
        }

    def get_detail(self, reserve_id: str, site_id: str | None = None) -> dict[str, Any]:
        df = self._base_frame()
        matches = df[df["sample_id"].astype(str) == reserve_id]
        if matches.empty:
            raise KeyError(reserve_id)
        row = matches.iloc[0]
        cell = self._cell(row)
        cell["top_contributors"] = self._shap_contributors(row)
        cell["site_id"] = site_id or self.settings.demo_site_id
        cell["geology"] = {
            "formation": str(row["formation"]),
            "elevation_m": round(float(row["elevation_m"]), 2),
            "slope_deg": round(float(row["slope_deg"]), 2),
            "aspect_deg": round(float(row["aspect_deg"]), 2),
            "depth_m": round(float(row["depth_m"]), 2),
        }
        return {**demo_envelope(), **cell}

    def boreholes(self, site_id: str | None = None, limit: int = 400) -> dict[str, Any]:
        df = self.store.boreholes()
        collars = (
            df.groupby("borehole_id", as_index=False)
            .agg(
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
                lithology=("lithology", lambda values: values.mode().iloc[0] if len(values.mode()) else values.iloc[0]),
                max_depth_m=("to_depth_m", "max"),
                mean_mn_pct=("mn_pct", "mean"),
            )
            .head(limit)
        )
        records = collars.round(4).to_dict(orient="records")
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "count": len(records),
            "boreholes": records,
        }

    def prediction_grid(
        self,
        site_id: str | None = None,
        bbox: str | None = None,
        cells_per_side: int = 10,
        coverage: float = 0.9,
    ) -> dict[str, Any]:
        bbox_values = self._parse_bbox(bbox) if bbox else None
        grid = generate_prediction_grid(
            self._model_dir(),
            bbox={
                "min_lat": bbox_values[1],
                "max_lat": bbox_values[3],
                "min_lon": bbox_values[0],
                "max_lon": bbox_values[2],
            }
            if bbox_values
            else None,
            cells_per_side=cells_per_side,
            coverage=coverage,
        )
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "bbox": [
                grid["bbox"]["min_lon"],
                grid["bbox"]["min_lat"],
                grid["bbox"]["max_lon"],
                grid["bbox"]["max_lat"],
            ],
            "cells_per_side": grid["cells_per_side"],
            "model_versions": grid["model_versions"],
            "context_method": grid["context_method"],
            "search_radius_m": grid["search_radius_m"],
            "cells": grid["cells"],
        }

    @staticmethod
    def _served_version_label(meta: dict[str, Any] | None) -> str:
        if meta and meta.get("model_name") and meta.get("version"):
            return f"{meta['model_name']}-{meta['version']}"
        return "reserve-prospectivity-legacy-artifact"

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.settings.require_model_artifacts and not reserve_prospectivity_available(self.settings):
            logger.warning("Reserve prediction requested in live mode without model artifacts")
            raise ModelUnavailableError(
                "Reserve prospectivity model is not available.",
                details={"model": "reserve_prospectivity"},
            )
        model_dir = self._model_dir()
        df = pd.DataFrame([payload])
        df["sample_id"] = "API-RESERVE-0001"
        df = ensure_spectral_indices(df)
        ensemble_path = resolve_ensemble_path(model_dir)
        version = self._served_version_label(self._model_metadata("prospectivity"))
        try:
            if ensemble_path is not None:
                scored = predict_ensemble_frame(df, model_dir)
                if scored is not None:
                    df["manganese_probability"] = scored["manganese_probability"]
                    df["prospectivity_class"] = scored["prospectivity_class"]
                    version = self._served_version_label(self._model_metadata("prospectivity_ensemble"))
                else:
                    raise FileNotFoundError("ensemble scoring returned None")
            else:
                scored = predict_prospectivity_frame(df, model_dir)
                df["manganese_probability"] = scored["manganese_probability"]
                df["prospectivity_class"] = scored["prospectivity_class"]
        except Exception as exc:
            logger.warning("Reserve prediction model unavailable; using demo heuristic fallback: %s", exc)
            if self.settings.require_model_artifacts:
                raise ModelUnavailableError(
                    "Reserve prospectivity model is not available.",
                    details={"model": "reserve_prospectivity"},
                ) from exc
            df["manganese_probability"] = heuristic_prospectivity(df)
            df["prospectivity_class"] = pd.cut(
                df["manganese_probability"],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
                include_lowest=True,
            ).astype(str)
            version = "reserve-prototype-heuristic-001"
        df["manganese_probability"] = df["manganese_probability"].astype(float).clip(0.01, 0.99)
        grade, grade_fn = predict_regressor_with_interval(df, model_dir, "grade")
        thickness, thickness_fn = predict_regressor_with_interval(df, model_dir, "thickness")
        df["predicted_grade_pct"] = grade.clip(2, 48).round(2) if grade is not None else self._predict_grade(df)
        df["predicted_thickness_m"] = thickness.clip(0.2, 18).round(2) if thickness is not None else self._predict_thickness(df)
        df["grade_interval_lower"] = df["grade_interval_upper"] = None
        df["thickness_interval_lower"] = df["thickness_interval_upper"] = None
        if grade is not None and grade_fn is not None:
            interval = grade_fn(grade)
            df["grade_interval_lower"] = round(interval["lower"], 2)
            df["grade_interval_upper"] = round(interval["upper"], 2)
        if thickness is not None and thickness_fn is not None:
            interval = thickness_fn(thickness)
            df["thickness_interval_lower"] = round(interval["lower"], 2)
            df["thickness_interval_upper"] = round(interval["upper"], 2)
        df["confidence"] = self._confidence(df)
        cell = self._cell(df.iloc[0])
        cell["data_support"]["model_version"] = version
        cell["data_support"]["model_algorithm"] = (
            self._model_metadata("prospectivity_ensemble")
            or self._model_metadata("prospectivity")
            or {}
        ).get("algorithm", "demo-heuristic")
        cell["data_support"]["prediction_type"] = "manganese_prospectivity"
        return {**demo_envelope(), "prediction": cell, "model_version": version}
