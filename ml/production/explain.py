from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.production.features import build_daily_features
from ml.production.forecast import _load_production_model


def rank_driver_impacts(driver_impacts: dict[str, float]) -> list[dict[str, float | str]]:
    total = sum(abs(value) for value in driver_impacts.values()) or 1.0
    ranked = sorted(driver_impacts.items(), key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "feature": feature,
            "direction": "negative" if value < 0 else "positive",
            "importance": round(abs(value) / total, 3),
        }
        for feature, value in ranked
    ]


def explain_production_latest(production: pd.DataFrame, model_dir: Path, limit: int = 5) -> list[dict]:
    artifact = model_dir / "production" / "forecast_xgboost.json"
    if not artifact.exists():
        return []
    try:
        model, columns = _load_production_model(str(artifact))
        features = build_daily_features(production)
        if features.empty:
            return []
        latest = features.iloc[[-1]].reindex(columns=columns, fill_value=0)
        try:
            import shap

            values = np.array(shap.TreeExplainer(model).shap_values(latest))
            shap_row = values[0] if values.ndim == 2 else values
            ranked = sorted(zip(columns, shap_row, strict=False), key=lambda item: abs(item[1]), reverse=True)
            total = sum(abs(value) for _, value in ranked) or 1.0
            return [
                {
                    "feature": feature,
                    "direction": "negative" if value < 0 else "positive",
                    "importance": round(abs(float(value)) / total, 3),
                    "value": round(float(latest.iloc[0][feature]), 3) if feature in latest.columns else None,
                }
                for feature, value in ranked[:limit]
            ]
        except Exception:
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                return []
            ranked = sorted(zip(columns, importances, strict=False), key=lambda item: item[1], reverse=True)
            return [
                {
                    "feature": feature,
                    "direction": "negative" if "downtime" in feature or "rainfall" in feature or "delay" in feature else "positive",
                    "importance": round(float(importance), 3),
                    "value": round(float(latest.iloc[0][feature]), 3) if feature in latest.columns else None,
                }
                for feature, importance in ranked[:limit]
            ]
    except Exception:
        return []
