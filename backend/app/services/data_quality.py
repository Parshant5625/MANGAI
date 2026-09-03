from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope


REQUIRED_COLUMNS = {
    "geological": {"sample_id", "latitude", "longitude", "elevation_m", "slope_deg", "aspect_deg", "depth_m", "formation"},
    "satellite_features": {"sample_id", "latitude", "longitude", "blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12"},
    "boreholes": {"borehole_id", "latitude", "longitude", "from_depth_m", "to_depth_m", "lithology"},
    "weather": {"date", "rainfall_mm", "soil_moisture", "temperature_c", "vegetation_index"},
    "equipment": {"date", "equipment_id", "equipment_type", "operating_hours", "downtime_hours", "utilization"},
    "blasting": {"date", "planned_blasts", "blasting_delay_hours", "delay_reason"},
    "production": {"date", "production_mt", "target_mt", "production_gap_mt"},
}


class DataQualityService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def run(self) -> dict[str, Any]:
        self.store.ensure_demo_data()
        synthetic = self.settings.resolved_data_dir / "synthetic"
        runs = []
        for name, columns in REQUIRED_COLUMNS.items():
            path = synthetic / f"{name}.csv"
            runs.append(self._check(path, name, columns))
        overall = round(float(sum(run["quality_score"] for run in runs) / max(len(runs), 1)), 3)
        return {**demo_envelope(), "overall_score": overall, "runs": runs}

    def _check(self, path: Path, name: str, required_columns: set[str]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        if not path.exists():
            return {
                "dataset_name": name,
                "source": "demo_synthetic",
                "row_count": 0,
                "missing_rate": 1.0,
                "duplicate_rate": 1.0,
                "schema_valid": False,
                "quality_score": 0.0,
                "details": {"missing_file": str(path)},
                "created_at": now,
            }
        df = pd.read_csv(path)
        missing_columns = sorted(required_columns - set(df.columns))
        missing_rate = float(df.isna().mean().mean()) if len(df) else 1.0
        duplicate_rate = float(df.duplicated().mean()) if len(df) else 1.0
        coordinate_ok = True
        if {"latitude", "longitude"}.issubset(df.columns):
            coordinate_ok = bool(df["latitude"].between(-90, 90).all() and df["longitude"].between(-180, 180).all())
        date_continuity = None
        freshness = None
        if "date" in df.columns and len(df):
            dates = pd.to_datetime(df["date"])
            span_days = int((dates.max() - dates.min()).days) + 1
            date_continuity = round(float(len(dates.drop_duplicates()) / max(span_days, 1)), 3)
            freshness = dates.max().date().isoformat()
        impossible_values = []
        if "mn_pct" in df.columns:
            impossible_values.append({"check": "mn_pct_range", "ok": bool(df["mn_pct"].between(0, 60).all())})
        if "rainfall_mm" in df.columns:
            impossible_values.append({"check": "rainfall_non_negative", "ok": bool((df["rainfall_mm"] >= 0).all())})
        if "utilization" in df.columns:
            impossible_values.append({"check": "utilization_0_1", "ok": bool(df["utilization"].between(0, 1.2).all())})
        outlier_rate = 0.0
        numeric = df.select_dtypes(include="number")
        if len(numeric) and len(numeric.columns):
            zscores = ((numeric - numeric.mean()) / (numeric.std(ddof=0).replace(0, 1))).abs()
            outlier_rate = float((zscores > 6).any(axis=1).mean())
        checks = {
            "schema": len(missing_columns) == 0,
            "missing": missing_rate <= 0.05,
            "duplicates": duplicate_rate <= 0.01,
            "coordinates": coordinate_ok,
            "date_continuity": date_continuity is None or date_continuity >= 0.95,
            "impossible_values": all(item["ok"] for item in impossible_values) if impossible_values else True,
            "outliers": outlier_rate <= 0.08,
        }
        score = sum(checks.values()) / len(checks)
        score = max(0.0, min(1.0, score - missing_rate * 0.5 - duplicate_rate * 0.25))
        return {
            "dataset_name": name,
            "source": "demo_synthetic",
            "row_count": int(len(df)),
            "missing_rate": round(missing_rate, 4),
            "duplicate_rate": round(duplicate_rate, 4),
            "schema_valid": len(missing_columns) == 0,
            "quality_score": round(float(score), 3),
            "details": {
                "missing_columns": missing_columns,
                "coordinate_range_ok": coordinate_ok,
                "date_continuity": date_continuity,
                "freshness": freshness,
                "impossible_values": impossible_values,
                "outlier_rate": round(outlier_rate, 4),
                "checks": checks,
            },
            "created_at": now,
        }

