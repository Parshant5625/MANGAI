from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope, heuristic_prospectivity
from ml.reserve.explain import explain_row
from ml.reserve.features import ensure_spectral_indices
from ml.reserve.inference import maybe_predict_regressor, predict_prospectivity_frame
from ml.reserve.resource_estimator import estimate_resource_potential


CELL_AREA_M2 = 10_000.0
DENSITY_T_PER_M3 = 3.6
_FRAME_CACHE: dict[tuple[str, float], pd.DataFrame] = {}


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
            model_dir / "reserve" / "prospectivity_xgboost.json",
            model_dir / "reserve_xgboost.json",
            model_dir / "reserve" / "grade_xgboost.json",
            model_dir / "reserve" / "thickness_xgboost.json",
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
        try:
            scored = predict_prospectivity_frame(df, self._model_dir())
            df["manganese_probability"] = scored["manganese_probability"]
            df["prospectivity_class"] = scored["prospectivity_class"]
        except FileNotFoundError:
            df["manganese_probability"] = heuristic_prospectivity(df)
            df["prospectivity_class"] = pd.cut(
                df["manganese_probability"],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
                include_lowest=True,
            ).astype(str)
        df["manganese_probability"] = df["manganese_probability"].astype(float).clip(0.01, 0.99)
        grade = maybe_predict_regressor(df, self._model_dir() / "reserve" / "grade_xgboost.json")
        thickness = maybe_predict_regressor(df, self._model_dir() / "reserve" / "thickness_xgboost.json")
        df["predicted_grade_pct"] = grade.clip(2, 48).round(2) if grade is not None else self._predict_grade(df)
        df["predicted_thickness_m"] = thickness.clip(0.2, 18).round(2) if thickness is not None else self._predict_thickness(df)
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

    def _resource_payload(self, probability: float, thickness_m: float, confidence: float, seed_key: str) -> dict[str, Any]:
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        estimate = estimate_resource_potential(
            probability=probability,
            thickness_m=thickness_m,
            cell_area_m2=CELL_AREA_M2,
            density_t_per_m3=DENSITY_T_PER_M3,
            probability_std=max(0.03, (1 - confidence) * 0.18),
            thickness_std_fraction=max(0.08, (1 - confidence) * 0.35),
            seed=seed,
        )
        return {
            "label": "prototype resource potential",
            "expected_tonnage": round(estimate.expected_tonnage, 2),
            "p10": round(estimate.p10, 2),
            "p50": round(estimate.p50, 2),
            "p90": round(estimate.p90, 2),
            "assumptions": {
                "cell_area_m2": CELL_AREA_M2,
                "estimated_density_t_per_m3": DENSITY_T_PER_M3,
                "uncertainty_method": "Monte Carlo",
                "classification_boundary": "prototype only, not official reserves",
            },
        }

    def _contributors(self, row: pd.Series) -> list[dict[str, Any]]:
        return [
            {"feature": "swir_ratio", "direction": "positive", "importance": 0.31, "value": round(float(row.get("swir_ratio", 0)), 3)},
            {"feature": "formation", "direction": "positive" if "Manganiferous" in str(row["formation"]) or "Gondite" in str(row["formation"]) else "neutral", "importance": 0.24, "value": str(row["formation"])},
            {"feature": "depth_m", "direction": "positive" if 10 <= float(row["depth_m"]) <= 55 else "neutral", "importance": 0.18, "value": round(float(row["depth_m"]), 2)},
        ]

    def _shap_contributors(self, row: pd.Series) -> list[dict[str, Any]]:
        model_path = self._model_dir() / "reserve" / "prospectivity_xgboost.json"
        if not model_path.exists():
            model_path = self._model_dir() / "reserve_xgboost.json"
        return explain_row(pd.DataFrame([row]), model_path) or self._contributors(row)

    def _cell(self, row: pd.Series) -> dict[str, Any]:
        resource = self._resource_payload(
            float(row["manganese_probability"]),
            float(row["predicted_thickness_m"]),
            float(row["confidence"]),
            str(row["sample_id"]),
        )
        return {
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
            "data_support": {
                "spectral_bands_present": int(
                    pd.Series(row[["blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12"]]).notna().sum()
                )
                if all(column in row.index for column in ["blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12"])
                else 0,
                "formation": str(row.get("formation", "")),
            },
        }

    def get_prospectivity(
        self,
        site_id: str | None = None,
        bbox: str | None = None,
        min_probability: float | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        df = self._base_frame()
        bbox_values: list[float] | None = None
        if bbox:
            bbox_values = [float(part) for part in bbox.split(",")]
            if len(bbox_values) != 4:
                raise ValueError("bbox must contain min_lon,min_lat,max_lon,max_lat")
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

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        df = pd.DataFrame([payload])
        df["sample_id"] = "API-RESERVE-0001"
        df = ensure_spectral_indices(df)
        try:
            scored = predict_prospectivity_frame(df, self._model_dir())
            df["manganese_probability"] = scored["manganese_probability"]
            df["prospectivity_class"] = scored["prospectivity_class"]
            version = "reserve-xgb-2026.09.001"
        except FileNotFoundError:
            df["manganese_probability"] = heuristic_prospectivity(df)
            df["prospectivity_class"] = pd.cut(
                df["manganese_probability"],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
                include_lowest=True,
            ).astype(str)
            version = "reserve-prototype-heuristic-001"
        grade = maybe_predict_regressor(df, self._model_dir() / "reserve" / "grade_xgboost.json")
        thickness = maybe_predict_regressor(df, self._model_dir() / "reserve" / "thickness_xgboost.json")
        df["predicted_grade_pct"] = grade.clip(2, 48).round(2) if grade is not None else self._predict_grade(df)
        df["predicted_thickness_m"] = thickness.clip(0.2, 18).round(2) if thickness is not None else self._predict_thickness(df)
        df["confidence"] = self._confidence(df)
        return {**demo_envelope(), "prediction": self._cell(df.iloc[0]), "model_version": version}
