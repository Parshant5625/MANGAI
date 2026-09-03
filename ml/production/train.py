from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from ml.common.metrics import mae, r2, rmse
from ml.common.registry import ModelVersionRecord, hash_file, hash_schema, write_registry_record
from ml.production.features import FEATURE_COLUMNS, build_daily_features, chronological_split


def train_production_model(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[2]
    production = pd.read_csv(root / "data/synthetic/production.csv", parse_dates=["date"])
    features = build_daily_features(production)
    train, validation, test = chronological_split(features)
    model = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["production_mt"])
    val_pred = model.predict(validation[FEATURE_COLUMNS])
    test_pred = model.predict(test[FEATURE_COLUMNS])
    naive_daily = float(pd.concat([train, validation])["production_mt"].tail(28).mean())
    naive = [naive_daily] * len(test)
    metrics = {
        "validation": "chronological_holdout",
        "val_mae": round(mae(validation["production_mt"], val_pred), 3),
        "val_rmse": round(rmse(validation["production_mt"], val_pred), 3),
        "val_r2": round(r2(validation["production_mt"], val_pred), 4),
        "test_mae": round(mae(test["production_mt"], test_pred), 3),
        "test_rmse": round(rmse(test["production_mt"], test_pred), 3),
        "test_r2": round(r2(test["production_mt"], test_pred), 4),
        "naive_test_mae": round(mae(test["production_mt"], naive), 3),
        "synthetic_data": True,
    }
    artifact_dir = root / "models" / "production"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / "forecast_xgboost.json"
    model.save_model(artifact)
    joblib.dump(FEATURE_COLUMNS, artifact_dir / "forecast_features.pkl")
    (artifact_dir / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    record = ModelVersionRecord(
        model_name="production_forecast",
        version="2026.09.001",
        task="time_series_forecast",
        algorithm="XGBoost",
        training_data_hash=hash_file(root / "data/synthetic/production.csv")[:16],
        feature_schema_hash=hash_schema({"features": FEATURE_COLUMNS})[:16],
        metrics=metrics,
        artifact_path="models/production/forecast_xgboost.json",
        status="candidate",
    )
    write_registry_record(record, root / "models" / "registry")
    return {"metrics": metrics, "artifact_path": record.artifact_path}


if __name__ == "__main__":
    print(train_production_model())
