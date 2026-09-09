from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.core.errors import ModelUnavailableError
from backend.app.services.demo_data import DemoDataStore, demo_envelope
from backend.app.services.model_artifacts import production_forecast_available
from ml.production.explain import explain_production_latest
from ml.production.forecast import conformal_width, naive_rolling_forecast, production_forecast, shortfall_probability
from ml.risk.shortfall import severity as risk_severity
from ml.risk.shortfall import shortfall_probability as fallback_shortfall_probability

logger = logging.getLogger(__name__)


class ProductionService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def _frame(self) -> pd.DataFrame:
        df = self.store.production().copy()
        df["gap_signed_mt"] = df["production_mt"] - df["target_mt"]
        return df

    def _driver_state(self, df: pd.DataFrame) -> dict[str, float]:
        recent = df.tail(7)
        history = df.tail(90)
        return {
            "rainfall_7d": float(recent["rainfall_mm"].sum()),
            "rainfall_7d_baseline": float(history["rainfall_mm"].rolling(7).sum().dropna().median()),
            "downtime_7d": float(recent["downtime_hours"].sum()),
            "downtime_7d_baseline": float(history["downtime_hours"].rolling(7).sum().dropna().median()),
            "blasting_delay_7d": float(recent["blasting_delay_hours"].sum()),
            "blasting_delay_7d_baseline": float(history["blasting_delay_hours"].rolling(7).sum().dropna().median()),
            "production_mean_7": float(recent["production_mt"].mean()),
            "production_mean_28": float(df.tail(28)["production_mt"].mean()),
            "target_mean_7": float(recent["target_mt"].mean()),
        }

    def forecast(self, site_id: str | None = None, horizon: int = 7) -> dict[str, Any]:
        horizon = max(1, min(int(horizon), 30))
        if self.settings.require_model_artifacts and not production_forecast_available(self.settings):
            raise ModelUnavailableError("Production forecast model is not available.", details={"model": "production_forecast"})
        df = self._frame()
        state = self._driver_state(df)
        latest_date = pd.Timestamp(df["date"].max())
        forecast_date = latest_date + pd.Timedelta(days=1)
        baseline_daily = state["production_mean_28"]
        rain_penalty = max(0.0, state["rainfall_7d"] - state["rainfall_7d_baseline"]) * 8.0
        downtime_penalty = max(0.0, state["downtime_7d"] - state["downtime_7d_baseline"]) * 42.0
        blast_penalty = max(0.0, state["blasting_delay_7d"] - state["blasting_delay_7d_baseline"]) * 65.0
        momentum = (state["production_mean_7"] - state["production_mean_28"]) * 0.35

        trained = production_forecast(df, self.settings.resolved_model_dir, horizon)
        if trained is None and self.settings.require_model_artifacts:
            raise ModelUnavailableError("Production forecast artifact is not available.", details={"horizon_days": horizon})
        daily_forecast = trained["forecast_daily_mt"] if trained else max(0.0, baseline_daily + momentum - rain_penalty - downtime_penalty - blast_penalty)
        trained_horizon = trained["trained_horizon_days"] if trained else None

        target_daily = float(df.tail(28)["target_mt"].mean())
        forecast_mt = float(daily_forecast * horizon)
        target_mt = float(target_daily * horizon)
        gap_mt = forecast_mt - target_mt

        interval_radius = conformal_width(self.settings.resolved_model_dir, horizon)
        if interval_radius is None:
            recent_error = float((df.tail(60)["production_mt"] - df.tail(60)["production_mt"].rolling(7).mean()).dropna().std())
            interval_radius = max(250.0, recent_error * math.sqrt(horizon))
        p10 = max(0.0, forecast_mt - interval_radius)
        p90 = forecast_mt + interval_radius

        probability_result = shortfall_probability(df, self.settings.resolved_model_dir, horizon)
        if probability_result is not None:
            probability, probability_horizon = probability_result
        else:
            rain_pressure = max(0.0, state["rainfall_7d"] - state["rainfall_7d_baseline"]) / 80
            downtime_pressure = max(0.0, state["downtime_7d"] - state["downtime_7d_baseline"]) / 80
            blast_pressure = max(0.0, state["blasting_delay_7d"] - state["blasting_delay_7d_baseline"]) / 20
            probability = fallback_shortfall_probability(gap_mt, target_mt, rain_pressure + downtime_pressure + blast_pressure)
            probability_horizon = None
        severity = risk_severity(probability, gap_mt, target_mt)
        shap_drivers = explain_production_latest(df, self.settings.resolved_model_dir)
        drivers = shap_drivers or self.top_drivers(state, rain_penalty, downtime_penalty, blast_penalty, momentum)

        horizon_series = [
            {
                "date": (forecast_date + pd.Timedelta(days=offset)).date().isoformat(),
                "horizon_day": offset + 1,
                "forecast_mt": round(float(daily_forecast), 2),
                "target_mt": round(target_daily, 2),
                "p10": round(max(0.0, daily_forecast - interval_radius / max(horizon, 1)), 2),
                "p90": round(daily_forecast + interval_radius / max(horizon, 1), 2),
            }
            for offset in range(horizon)
        ]
        model_version = (
            f"production-{trained_horizon}d-trained"
            if trained_horizon is not None
            else "production-demo-chronological-baseline-001"
        )
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "forecast_date": forecast_date.date().isoformat(),
            "forecast_origin": latest_date.date().isoformat(),
            "horizon_days": horizon,
            "forecast_mt": round(forecast_mt, 2),
            "target_mt": round(target_mt, 2),
            "gap_mt": round(gap_mt, 2),
            "shortfall_probability": round(float(probability), 3),
            "severity": severity,
            "prediction_interval": {"p10": round(p10, 2), "p50": round(forecast_mt, 2), "p90": round(p90, 2)},
            "horizon_series": horizon_series,
            "top_drivers": drivers,
            "model_version": model_version,
            "baseline_forecast_mt": round(float(naive_rolling_forecast(df, horizon_days=horizon)), 2),
            "data_freshness": {
                "latest_production_date": latest_date.date().isoformat(),
                "records": int(len(df)),
                "demo_snapshot": True,
                "future_exogenous_assumption": "last_observed_context_held_constant" if trained_horizon else "not_applicable",
                "forecast_model_horizon_days": trained_horizon,
                "shortfall_model_horizon_days": probability_horizon,
            },
        }

    def risk(self, site_id: str | None = None) -> dict[str, Any]:
        return self.forecast(site_id=site_id, horizon=7)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        horizon = int(payload.get("horizon_days", 7))
        target_mt = float(payload.get("target_mt", 8500)) * horizon
        base = self.forecast(horizon=horizon)
        penalty = float(payload.get("rainfall_mm_7d", 0)) * 8 + float(payload.get("downtime_hours_7d", 0)) * 42 + float(payload.get("blasting_delay_7d", 0)) * 65
        forecast_mt = max(0.0, base["baseline_forecast_mt"] - penalty)
        gap_mt = forecast_mt - target_mt
        probability = fallback_shortfall_probability(gap_mt, target_mt, float(payload.get("rainfall_mm_7d", 0)) / 80 + float(payload.get("downtime_hours_7d", 0)) / 80 + float(payload.get("blasting_delay_7d", 0)) / 20)
        base.update({"forecast_mt": round(forecast_mt, 2), "target_mt": round(target_mt, 2), "gap_mt": round(gap_mt, 2), "shortfall_probability": round(probability, 3), "severity": risk_severity(probability, gap_mt, target_mt)})
        return base

    def top_drivers(self, state: dict[str, float], rain_penalty: float, downtime_penalty: float, blast_penalty: float, momentum: float) -> list[dict[str, Any]]:
        raw = [
            ("downtime_hours_7d", downtime_penalty, state["downtime_7d"], "negative"),
            ("rainfall_7d", rain_penalty, state["rainfall_7d"], "negative"),
            ("blasting_delay_7d", blast_penalty, state["blasting_delay_7d"], "negative"),
            ("production_momentum_7d", abs(momentum), round(momentum, 2), "positive" if momentum >= 0 else "negative"),
        ]
        total = sum(value for _, value, _, _ in raw) or 1.0
        return [{"feature": feature, "direction": direction, "importance": round(value / total, 3), "value": metric} for feature, value, metric, direction in sorted(raw, key=lambda item: item[1], reverse=True)]

    def history(self, days: int = 60) -> list[dict[str, Any]]:
        df = self._frame().tail(days).copy()
        df["date"] = df["date"].dt.date.astype(str)
        return df[["date", "production_mt", "target_mt", "gap_signed_mt", "rainfall_mm", "downtime_hours", "blasting_delay_hours"]].round(2).to_dict(orient="records")
