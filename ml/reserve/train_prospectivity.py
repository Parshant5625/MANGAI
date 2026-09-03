from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from ml.common.registry import ModelVersionRecord, hash_file, hash_schema, write_registry_record
from ml.reserve.features import (
    LEAKAGE_EXCLUSIONS,
    load_fused_reserve_table,
    prepare_reserve_matrix,
    project_root,
)
from ml.reserve.spatial import classification_metrics, grouped_cv_scores, regression_metrics, spatial_holdout_indices


def _artifact_dirs(root: Path) -> tuple[Path, Path, Path]:
    model_dir = root / "models"
    reserve_dir = model_dir / "reserve"
    registry_dir = model_dir / "registry"
    reserve_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, reserve_dir, registry_dir


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_prospectivity(root: Path | None = None) -> dict:
    root = root or project_root()
    df = load_fused_reserve_table(root)
    X = prepare_reserve_matrix(df)
    y = df["is_manganese"].astype(int)
    train_idx, test_idx, groups = spatial_holdout_indices(df)
    model = XGBClassifier(
        n_estimators=280,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = classification_metrics(y.iloc[test_idx], probabilities)
    metrics.update(
        grouped_cv_scores(
            lambda: XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            X,
            y,
            groups,
            "classification",
        )
    )
    metrics["validation"] = "spatial_block_holdout"
    metrics["held_out_blocks"] = int(groups.iloc[test_idx].nunique())
    metrics["leakage_exclusions"] = LEAKAGE_EXCLUSIONS
    metrics["synthetic_data"] = True
    return _persist_model(
        root,
        model_name="reserve_prospectivity",
        task="binary_classification",
        model=model,
        feature_columns=list(X.columns),
        metrics=metrics,
        legacy_stem="reserve_xgboost",
    )


def train_grade(root: Path | None = None) -> dict:
    return _train_regressor(root, target="mn_pct", model_name="reserve_grade", artifact_stem="grade_xgboost")


def train_thickness(root: Path | None = None) -> dict:
    return _train_regressor(root, target="ore_thickness_m", model_name="reserve_thickness", artifact_stem="thickness_xgboost")


def _train_regressor(root: Path | None, target: str, model_name: str, artifact_stem: str) -> dict:
    root = root or project_root()
    df = load_fused_reserve_table(root)
    X = prepare_reserve_matrix(df)
    y = df[target].astype(float)
    train_idx, test_idx, groups = spatial_holdout_indices(df)
    model = XGBRegressor(
        n_estimators=280,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    metrics = regression_metrics(y.iloc[test_idx], preds)
    metrics.update(
        grouped_cv_scores(
            lambda: XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
            X,
            y,
            groups,
            "regression",
        )
    )
    metrics["validation"] = "spatial_block_holdout"
    metrics["target"] = target
    metrics["leakage_exclusions"] = LEAKAGE_EXCLUSIONS
    metrics["synthetic_data"] = True
    return _persist_model(
        root,
        model_name=model_name,
        task="regression",
        model=model,
        feature_columns=list(X.columns),
        metrics=metrics,
        legacy_stem=None,
        artifact_stem=artifact_stem,
    )


def _persist_model(
    root: Path,
    model_name: str,
    task: str,
    model,
    feature_columns: list[str],
    metrics: dict,
    legacy_stem: str | None,
    artifact_stem: str | None = None,
) -> dict:
    model_dir, reserve_dir, registry_dir = _artifact_dirs(root)
    stem = artifact_stem or "prospectivity_xgboost"
    artifact = reserve_dir / f"{stem}.json"
    model.save_model(artifact)
    joblib.dump(feature_columns, reserve_dir / f"{stem}_features.pkl")
    _save_json(reserve_dir / f"{stem}_metrics.json", metrics)
    if legacy_stem:
        model.save_model(model_dir / f"{legacy_stem}.json")
        joblib.dump(feature_columns, model_dir / f"{legacy_stem.replace('xgboost', 'features')}.pkl")
        _save_json(model_dir / "reserve_metrics.json", metrics)
        importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values(
            "importance",
            ascending=False,
        )
        importance.to_csv(model_dir / "reserve_feature_importance.csv", index=False)
    record = ModelVersionRecord(
        model_name=model_name,
        version="2026.09.001",
        task=task,
        algorithm="XGBoost",
        training_data_hash=hash_file(root / "data/synthetic/geological.csv")[:16],
        feature_schema_hash=hash_schema({"features": feature_columns})[:16],
        metrics=metrics,
        artifact_path=str(artifact.relative_to(root)).replace("\\", "/"),
        status="candidate",
    )
    write_registry_record(record, registry_dir)
    return {"model_name": model_name, "metrics": metrics, "artifact_path": record.artifact_path}


if __name__ == "__main__":
    print(train_prospectivity())
