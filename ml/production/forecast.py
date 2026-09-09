from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.production.features import FEATURE_COLUMNS, build_daily_features
from ml.production.pipeline import HORIZONS


def naive_rolling_forecast(production: pd.DataFrame, horizon_days: int = 7, window: int = 28) -> float:
    df = production.copy().sort_values("date")
    return float(df["production_mt"].tail(window).mean()) * horizon_days


def supported_horizon(horizon_days: int) -> int:
    return min(HORIZONS, key=lambda h: abs(h - int(horizon_days)))


@lru_cache(maxsize=16)
def _load_model(path: str):
    return joblib.load(path)


def _latest_features(production: pd.DataFrame) -> pd.DataFrame:
    features = build_daily_features(production)
    if features.empty:
        return features
    return features.iloc[[-1]].reindex(columns=FEATURE_COLUMNS)


def xgb_daily_forecast(production: pd.DataFrame, model_dir: Path) -> float | None:
    path = model_dir / "production" / "forecast_1d.pkl"
    if not path.exists():
        return None
    features = _latest_features(production)
    if features.empty:
        return None
    return max(0.0, float(_load_model(str(path)).predict(features)[0]))


def production_forecast(production: pd.DataFrame, model_dir: Path, horizon_days: int) -> dict | None:
    trained_horizon = supported_horizon(horizon_days)
    path = model_dir / "production" / f"forecast_{trained_horizon}d.pkl"
    if not path.exists():
        return None
    features = _latest_features(production)
    if features.empty:
        return None
    model = _load_model(str(path))
    predicted_daily = max(0.0, float(model.predict(features)[0]))
    return {"forecast_daily_mt": predicted_daily, "trained_horizon_days": trained_horizon}


def shortfall_probability(production: pd.DataFrame, model_dir: Path, horizon_days: int) -> tuple[float, int] | None:
    trained_horizon = supported_horizon(horizon_days)
    path = model_dir / "production" / f"shortfall_{trained_horizon}d.pkl"
    if not path.exists():
        return None
    features = _latest_features(production)
    if features.empty:
        return None
    probability = float(_load_model(str(path)).predict_proba(features)[:, 1][0])
    return max(0.0, min(1.0, probability)), trained_horizon


def conformal_width(model_dir: Path, horizon_days: int) -> float | None:
    """Read the validation-only conformal radius from the training report."""
    report = model_dir / "production" / "training_report.json"
    if not report.exists():
        return None
    import json

    payload = json.loads(report.read_text(encoding="utf-8"))
    trained_horizon = str(supported_horizon(horizon_days))
    model_metrics = payload.get("models", {}).get(trained_horizon, {})
    value = model_metrics.get("conformal_quantile_90")
    return float(value) if value is not None else None
