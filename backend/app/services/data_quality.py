from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope
from ml.common.contracts import (
    BLASTING_CONTRACT,
    BOREHOLE_CONTRACT,
    EQUIPMENT_CONTRACT,
    GEOLOGICAL_CONTRACT,
    PRODUCTION_CONTRACT,
    SATELLITE_CONTRACT,
    WEATHER_CONTRACT,
)
from ml.common.validation import validate_borehole_intervals

# Canonical schema definitions come from the data contracts, not a local copy.
DATASET_CONTRACTS = {
    GEOLOGICAL_CONTRACT.name: GEOLOGICAL_CONTRACT,
    SATELLITE_CONTRACT.name: SATELLITE_CONTRACT,
    BOREHOLE_CONTRACT.name: BOREHOLE_CONTRACT,
    WEATHER_CONTRACT.name: WEATHER_CONTRACT,
    EQUIPMENT_CONTRACT.name: EQUIPMENT_CONTRACT,
    BLASTING_CONTRACT.name: BLASTING_CONTRACT,
    PRODUCTION_CONTRACT.name: PRODUCTION_CONTRACT,
}


class DataQualityService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def run(self) -> dict[str, Any]:
        self.store.ensure_demo_data()
        synthetic = self.settings.resolved_data_dir / "synthetic"
        runs = []
        for name, contract in DATASET_CONTRACTS.items():
            path = synthetic / f"{name}.csv"
            runs.append(self._check(path, name, contract))
        overall = round(float(sum(run["quality_score"] for run in runs) / max(len(runs), 1)), 3)
        return {**demo_envelope(), "overall_score": overall, "runs": runs}

    def _check(self, path: Path, name: str, contract) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        if not path.exists():
            return {
                "dataset_name": name,
                "source": contract.source,
                "row_count": 0,
                "missing_rate": 1.0,
                "duplicate_rate": 1.0,
                "schema_valid": False,
                "quality_score": 0.0,
                "details": {"missing_file": str(path)},
                "created_at": now,
            }
        df = pd.read_csv(path)
        required_columns = set(contract.required_columns())
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

        # Contract-driven range checks (no hard-coded values).
        range_violations = []
        for column in contract.columns:
            if column.name not in df.columns or column.dtype not in ("float", "int"):
                continue
            series = df[column.name].dropna()
            if series.empty:
                continue
            if column.min_value is not None and (series < column.min_value).any():
                range_violations.append(column.name)
            if column.max_value is not None and (series > column.max_value).any():
                range_violations.append(column.name)
        range_ok = not range_violations

        # Dataset-specific structural checks.
        interval_ok = True
        if name == "boreholes":
            interval_result = validate_borehole_intervals(df)
            interval_ok = interval_result.valid

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
            "contract_ranges": range_ok,
            "structural": interval_ok,
            "outliers": outlier_rate <= 0.08,
        }
        score = sum(checks.values()) / len(checks)
        score = max(0.0, min(1.0, score - missing_rate * 0.5 - duplicate_rate * 0.25))
        return {
            "dataset_name": name,
            "source": contract.source,
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
                "contract_range_violations": range_violations,
                "outlier_rate": round(outlier_rate, 4),
                "checks": checks,
            },
            "created_at": now,
        }
