from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from ml.production.features import FEATURE_COLUMNS, build_daily_features


def naive_rolling_forecast(production: pd.DataFrame, horizon_days: int = 7, window: int = 28) -> float:
    df = production.copy().sort_values("date")
    daily = float(df["production_mt"].tail(window).mean())
    return daily * horizon_days


@lru_cache(maxsize=2)
def _load_production_model(path: str) -> tuple[XGBRegressor, list[str]]:
    model = XGBRegressor()
    model.load_model(path)
    columns = joblib.load(Path(path).with_name("forecast_features.pkl"))
    return model, columns


def _predict_latest(features: pd.DataFrame, model: XGBRegressor, columns: list[str]) -> float:
    latest = features.iloc[[-1]].reindex(columns=columns, fill_value=0)
    return float(model.predict(latest)[0])


def xgb_daily_forecast(production: pd.DataFrame, model_dir: Path) -> float | None:
    artifact = model_dir / "production" / "forecast_xgboost.json"
    if not artifact.exists():
        return None
    model, columns = _load_production_model(str(artifact))
    features = build_daily_features(production)
    if features.empty:
        return None
    return _predict_latest(features, model, columns)


def xgb_horizon_forecast(production: pd.DataFrame, model_dir: Path, horizon_days: int) -> list[dict] | None:
    artifact = model_dir / "production" / "forecast_xgboost.json"
    if not artifact.exists():
        return None
    model, columns = _load_production_model(str(artifact))
    history = production.copy().sort_values("date")
    last = history.iloc[-1]
    target_daily = float(history["target_mt"].tail(28).mean())
    series = []
    for step in range(horizon_days):
        features = build_daily_features(history)
        if features.empty:
            return None
        daily = max(0.0, _predict_latest(features, model, columns))
        next_date = pd.Timestamp(history["date"].max()) + pd.Timedelta(days=1)
        next_row = last.copy()
        next_row["date"] = next_date
        next_row["production_mt"] = daily
        next_row["target_mt"] = target_daily
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        series.append(
            {
                "date": next_date.date().isoformat(),
                "horizon_day": step + 1,
                "forecast_mt": round(daily, 2),
                "target_mt": round(target_daily, 2),
            }
        )
    return series


def shortfall_gap(forecast_mt: float, target_mt: float) -> float:
    return forecast_mt - target_mt
