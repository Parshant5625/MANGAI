from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.app.core.config import get_settings


BOUNDARY_NOTICE = (
    "DEMO / SYNTHETIC DATA. MANGAI is a decision-support prototype until validated "
    "with MOIL data, domain experts, applicable regulations, and operational systems."
)


class DemoDataStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.root = settings.project_root
        self.data_dir = settings.resolved_data_dir
        self.model_dir = settings.resolved_model_dir
        self.synthetic_dir = self.data_dir / "synthetic"
        self.processed_dir = self.data_dir / "processed"

    def ensure_demo_data(self) -> None:
        required = [
            self.synthetic_dir / "geological.csv",
            self.synthetic_dir / "satellite_features.csv",
            self.synthetic_dir / "production.csv",
            self.synthetic_dir / "equipment.csv",
            self.synthetic_dir / "weather.csv",
            self.synthetic_dir / "blasting.csv",
        ]
        if all(path.exists() for path in required):
            return
        from ml.generate_data import main as generate_demo_data

        generate_demo_data()

    def read_csv(self, relative_path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        self.ensure_demo_data()
        path = self.root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required demo dataset not found: {path}")
        return pd.read_csv(path, parse_dates=parse_dates)

    @lru_cache(maxsize=1)
    def geological(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/geological.csv")

    @lru_cache(maxsize=1)
    def satellite(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/satellite_features.csv")

    @lru_cache(maxsize=1)
    def boreholes(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/boreholes.csv")

    @lru_cache(maxsize=1)
    def production(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/production.csv", parse_dates=["date"]).sort_values("date")

    @lru_cache(maxsize=1)
    def equipment(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/equipment.csv", parse_dates=["date"]).sort_values("date")

    @lru_cache(maxsize=1)
    def weather(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/weather.csv", parse_dates=["date"]).sort_values("date")

    @lru_cache(maxsize=1)
    def blasting(self) -> pd.DataFrame:
        return self.read_csv("data/synthetic/blasting.csv", parse_dates=["date"]).sort_values("date")

    @lru_cache(maxsize=1)
    def reserve_predictions(self) -> pd.DataFrame:
        path = self.processed_dir / "reserve_predictions.csv"
        if path.exists():
            return pd.read_csv(path)
        geological = self.geological()
        satellite = self.satellite()
        merged = geological.merge(satellite, on=["sample_id", "latitude", "longitude"], how="inner")
        merged["manganese_probability"] = heuristic_prospectivity(merged)
        merged["prospectivity_class"] = pd.cut(
            merged["manganese_probability"],
            bins=[0, 0.25, 0.5, 0.75, 1],
            labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
            include_lowest=True,
        ).astype(str)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        return merged


def normalize(series: pd.Series, default: float = 0.5) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value == min_value:
        return pd.Series(default, index=series.index)
    return (series - min_value) / (max_value - min_value)


def heuristic_prospectivity(df: pd.DataFrame) -> pd.Series:
    formation_signal = df.get("formation", pd.Series("", index=df.index)).astype(str).str.contains(
        "Manganiferous|Gondite|Laterite", case=False, regex=True
    )
    depth = df.get("depth_m", pd.Series(35, index=df.index))
    depth_window = ((depth >= 10) & (depth <= 55)).astype(float)
    slope = normalize(df.get("slope_deg", pd.Series(15, index=df.index)))
    swir = normalize(df.get("swir_ratio", pd.Series(1.2, index=df.index)))
    bsi = normalize(df.get("bare_soil_index", pd.Series(0, index=df.index)))
    ndvi = normalize(df.get("ndvi", pd.Series(0.3, index=df.index)))
    score = 0.22 + 0.26 * formation_signal.astype(float) + 0.18 * depth_window
    score += 0.2 * swir + 0.12 * bsi + 0.08 * (1 - ndvi) - 0.08 * slope
    return score.clip(0.01, 0.99)


def demo_envelope() -> dict[str, object]:
    settings = get_settings()
    return {
        "data_mode": settings.data_mode,
        "synthetic_data": settings.data_mode == "demo",
        "boundary_notice": BOUNDARY_NOTICE,
    }

