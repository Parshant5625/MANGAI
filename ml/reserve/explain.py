from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.reserve.features import prepare_reserve_matrix
from ml.reserve.inference import load_booster_and_columns


def top_contributors_from_gain(model, feature_columns: list[str], row: pd.Series, limit: int = 5) -> list[dict]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ranked = sorted(zip(feature_columns, importances, strict=False), key=lambda item: item[1], reverse=True)
    contributors = []
    for feature, importance in ranked[:limit]:
        value = row.get(feature.replace("formation_", ""), row.get(feature))
        direction = "positive" if importance >= 0 else "negative"
        contributors.append(
            {
                "feature": feature,
                "direction": direction,
                "importance": round(float(importance), 3),
                "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else value,
            }
        )
    return contributors


def explain_row(df: pd.DataFrame, model_path: Path, task: str = "classification", limit: int = 5) -> list[dict]:
    if not model_path.exists():
        return []
    try:
        model, columns = load_booster_and_columns(model_path, task=task)
    except Exception:
        return []
    features = prepare_reserve_matrix(df, columns)
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = np.array(explainer.shap_values(features.iloc[[0]]))
        if values.ndim > 2:
            values = values[1] if values.shape[0] > 1 else values[0]
        shap_row = values[0]
        ranked = sorted(zip(columns, shap_row, strict=False), key=lambda item: abs(item[1]), reverse=True)
        total = sum(abs(value) for _, value in ranked) or 1.0
        return [
            {
                "feature": feature,
                "direction": "positive" if value >= 0 else "negative",
                "importance": round(abs(float(value)) / total, 3),
                "value": round(float(features.iloc[0][feature]), 4) if feature in features.columns else None,
            }
            for feature, value in ranked[:limit]
        ]
    except Exception:
        return top_contributors_from_gain(model, columns, df.iloc[0], limit=limit)
