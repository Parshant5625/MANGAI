from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.adapters.satellite.fusion_pipeline import LiveSatelliteFusionPipeline
from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError, ModelUnavailableError
from ml.common.provenance import DataProvenance
from ml.reserve.inference import (
    load_model_metadata,
    load_support_assessor,
    predict_ensemble_frame,
    predict_regressor_with_interval,
    resolve_ensemble_path,
    resolve_model_path,
)
from ml.reserve.live_fusion import GEOLOGY_CONTEXT_COLUMNS, fuse_live_satellite_with_geology
from ml.reserve.resource_estimator import estimate_resource_potential_with_intervals

CELL_AREA_M2 = 10_000.0
DENSITY_T_PER_M3 = 3.6


class LiveSatelliteReserveService:
    """Run reserve inference from real satellite data without silent fallback."""

    def __init__(self, pipeline: LiveSatelliteFusionPipeline | None = None) -> None:
        self.settings = get_settings()
        self.pipeline = pipeline or LiveSatelliteFusionPipeline()

    def _geology_path(self) -> Path:
        return self.settings.resolved_data_dir / "raw" / "geological.csv"

    def _load_geology(self, path: Path | None = None) -> tuple[pd.DataFrame, DataProvenance]:
        path = path or self._geology_path()
        if not path.exists():
            raise DataUnavailableError(
                "Live geological context file is unavailable.",
                details={"path": str(path), "required_columns": list(GEOLOGY_CONTEXT_COLUMNS)},
            )
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise DataUnavailableError("Live geological context could not be read.", details={"path": str(path)}) from exc
        missing = [column for column in GEOLOGY_CONTEXT_COLUMNS if column not in frame.columns]
        if missing:
            raise DataUnavailableError(
                "Geological context failed the target-free context contract.",
                details={"missing_columns": missing, "required_columns": list(GEOLOGY_CONTEXT_COLUMNS)},
            )
        if frame["sample_id"].astype(str).duplicated().any():
            raise DataUnavailableError("Geological context contains duplicate sample_id values.")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        path_parts = {part.lower() for part in path.parts}
        is_synthetic = "synthetic" in path_parts and "data" in path_parts
        provenance = DataProvenance(
            source_name="synthetic_geological_demo" if is_synthetic else "live_geological_file",
            source_kind="synthetic" if is_synthetic else "local_file",
            mode="demo" if is_synthetic else "live",
            dataset="geological",
            acquired_at=None,
            ingested_at=None,
            source_version=None,
            source_uri=str(path),
            checksum=checksum,
            license_note=(
                "Synthetic geological context used only to demonstrate the live satellite path."
                if is_synthetic
                else "Operator-supplied geological context; validate licensing and provenance before production use."
            ),
            quality_score=1.0,
            row_count=len(frame),
        )
        return frame, provenance

    @staticmethod
    def _version(model_dir: Path, name: str) -> str:
        path = resolve_model_path(model_dir, name)
        if path is None:
            return "unknown"
        metadata = load_model_metadata(path)
        if metadata and metadata.get("version"):
            return str(metadata["version"])
        return path.name

    @staticmethod
    def _classify(probability: float) -> str:
        if probability >= 0.75:
            return "VERY_HIGH" if probability >= 0.85 else "HIGH"
        if probability >= 0.5:
            return "MODERATE"
        return "LOW"

    def _resource(self, row: pd.Series, index: int) -> dict[str, Any]:
        low = row.get("thickness_interval_lower")
        high = row.get("thickness_interval_upper")
        thickness = float(row["predicted_thickness_m"])
        low = float(low) if pd.notna(low) else thickness
        high = float(high) if pd.notna(high) else thickness
        seed_key = f"{row['sample_id']}:{index}:{row['satellite_provenance_checksum']}"
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        estimate = estimate_resource_potential_with_intervals(
            probability=float(row["manganese_probability"]),
            thickness_point=thickness,
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

    def _cell(self, row: pd.Series, index: int, model_version: str) -> dict[str, Any]:
        distance = float(row["geology_match_distance_m"])
        spectral_columns = ["blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12", "land_surface_temperature"]
        completeness = float(row[spectral_columns].notna().mean())
        distance_score = max(0.0, 1.0 - distance / 500.0)
        confidence = round(max(0.0, min(1.0, 0.65 * completeness + 0.35 * distance_score)), 2)
        cell: dict[str, Any] = {
            "id": str(row["sample_id"]),
            "latitude": round(float(row["latitude"]), 6),
            "longitude": round(float(row["longitude"]), 6),
            "probability": round(float(row["manganese_probability"]), 4),
            "prospectivity_class": self._classify(float(row["manganese_probability"])),
            "predicted_grade_pct": round(float(row["predicted_grade_pct"]), 2),
            "predicted_thickness_m": round(float(row["predicted_thickness_m"]), 2),
            "confidence": confidence,
            "resource_potential": self._resource(row, index),
            "top_contributors": [],
            "data_support": {
                "mode": "live",
                "satellite_source": str(row["satellite_source"]),
                "satellite_acquired_at": row.get("satellite_acquired_at"),
                "satellite_provenance_checksum": str(row["satellite_provenance_checksum"]),
                "geology_source": "synthetic_demo" if str(row["geology_source"]) == "synthetic_demo" else "operator_supplied_local_file",
                "geology_match_distance_m": round(distance, 2),
                "feature_completeness": round(completeness, 3),
                "model_version": model_version,
                "boundary": "prototype resource potential; not an official mineral resource or reserve",
            },
        }
        if pd.notna(row.get("grade_interval_lower")) and pd.notna(row.get("grade_interval_upper")):
            cell["grade_interval"] = {
                "lower": round(float(row["grade_interval_lower"]), 2),
                "upper": round(float(row["grade_interval_upper"]), 2),
                "level": 0.9,
                "method": "split_conformal",
                "coverage": 0.9,
                "non_negative": False,
            }
        if pd.notna(row.get("thickness_interval_lower")) and pd.notna(row.get("thickness_interval_upper")):
            cell["thickness_interval"] = {
                "lower": round(float(row["thickness_interval_lower"]), 2),
                "upper": round(float(row["thickness_interval_upper"]), 2),
                "level": 0.9,
                "method": "split_conformal",
                "coverage": 0.9,
                "non_negative": True,
            }
        support = load_support_assessor(self.settings.resolved_model_dir)
        if support is not None:
            assessment = support.assess(
                {name: float(row[name]) for name in ["elevation_m", "slope_deg", "depth_m", "ndvi", "ndwi", "swir_ratio"] if name in row and pd.notna(row[name])},
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )
            cell["extrapolation"] = assessment.get("extrapolation")
            cell["data_support_detail"] = assessment.get("data_support")
        return cell

    def predict(
        self,
        *,
        site_id: str,
        start: str,
        end: str,
        latitude: float,
        longitude: float,
        max_temporal_days: int = 16,
        max_geology_distance_m: float = 500.0,
        limit: int = 5,
        geology_path: Path | None = None,
    ) -> dict[str, Any]:
        if self.settings.data_mode != "live":
            raise DataUnavailableError("Live satellite reserve inference requires DATA_MODE=live.")
        model_dir = self.settings.resolved_model_dir
        if resolve_ensemble_path(model_dir) is None and resolve_model_path(model_dir, "prospectivity") is None:
            raise ModelUnavailableError("Reserve prospectivity model artifact is not available.", details={"model": "reserve_prospectivity"})
        if resolve_model_path(model_dir, "grade") is None or resolve_model_path(model_dir, "thickness") is None:
            raise ModelUnavailableError("Reserve grade/thickness model artifacts are not available.", details={"models": ["grade", "thickness"]})

        geology, geology_provenance = self._load_geology(geology_path)
        fusion = self.pipeline.run(
            site_id=site_id,
            start=start,
            end=end,
            latitude=latitude,
            longitude=longitude,
            max_temporal_days=max_temporal_days,
            limit=limit,
        )
        fused = fuse_live_satellite_with_geology(
            fusion.batch,
            geology,
            max_distance_m=max_geology_distance_m,
        )
        if fused.data.empty:
            raise DataUnavailableError(
                "No live satellite pixels matched the geological context within the configured distance.",
                details={"max_geology_distance_m": max_geology_distance_m, "unmatched_satellite": fused.unmatched_satellite},
            )

        frame = fused.data.copy()
        frame["geology_source"] = geology_provenance.source_name
        ensemble_path = resolve_ensemble_path(model_dir)
        if ensemble_path is not None:
            scored = predict_ensemble_frame(frame, model_dir)
            if scored is None:
                raise ModelUnavailableError("Reserve ensemble artifact could not be loaded.", details={"model": "reserve_prospectivity_ensemble"})
        else:
            from ml.reserve.inference import predict_prospectivity_frame
            scored = predict_prospectivity_frame(frame, model_dir)
        frame["manganese_probability"] = scored["manganese_probability"]
        frame["prospectivity_class"] = scored["prospectivity_class"]
        grade, grade_fn = predict_regressor_with_interval(frame, model_dir, "grade")
        thickness, thickness_fn = predict_regressor_with_interval(frame, model_dir, "thickness")
        if grade is None or thickness is None:
            raise ModelUnavailableError("Reserve grade/thickness inference failed.")
        frame["predicted_grade_pct"] = grade.clip(2, 48)
        frame["predicted_thickness_m"] = thickness.clip(0.2, 18)
        frame["grade_interval_lower"] = frame["grade_interval_upper"] = None
        frame["thickness_interval_lower"] = frame["thickness_interval_upper"] = None
        if grade_fn is not None:
            interval = grade_fn(grade)
            frame["grade_interval_lower"] = interval["lower"]
            frame["grade_interval_upper"] = interval["upper"]
        if thickness_fn is not None:
            interval = thickness_fn(thickness)
            frame["thickness_interval_lower"] = interval["lower"]
            frame["thickness_interval_upper"] = interval["upper"]

        model_version = self._version(model_dir, "prospectivity")
        synthetic_geology = geology_provenance.mode == "demo"
        return {
            "data_mode": "live",
            "synthetic_data": synthetic_geology,
            "mixed_data": synthetic_geology,
            "boundary_notice": "Live-source prototype resource potential; not an official mineral resource or reserve.",
            "site_id": site_id,
            "count": len(frame),
            "matched_geological_context": fused.matched,
            "unmatched_satellite": fused.unmatched_satellite,
            "geology_provenance": geology_provenance.to_dict(),
            "satellite_provenance": fusion.batch.provenance.to_dict(),
            "sentinel_scene_count": fusion.sentinel_scene_count,
            "thermal_scene_count": fusion.thermal_scene_count,
            "temporal_distance_days": list(fusion.temporal_distance_days),
            "cells": [self._cell(row, index, model_version) for index, (_, row) in enumerate(frame.iterrows())],
        }
