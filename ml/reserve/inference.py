from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from ml.reserve.features import ensure_spectral_indices, prepare_reserve_matrix


@lru_cache(maxsize=8)
def _load_booster(path: str, task: str):
    model = XGBClassifier() if task == "classification" else XGBRegressor()
    model.load_model(path)
    return model


def _feature_path(model_path: Path) -> Path:
    name = model_path.name
    if name in ("reserve_xgboost.json", "reserve_model.joblib"):
        return model_path.with_name("reserve_features.pkl")
    stem = model_path.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return model_path.with_name(stem + "_features.pkl")


def resolve_model_path(model_dir: Path, name: str) -> Path | None:
    """Resolve the current serving artifact for a reserve model, if any.

    Prefers the joblib bundle written by the Phase 3 training pipeline and
    falls back to the legacy XGBoost-native artifacts.
    """
    candidates = [
        model_dir / "reserve" / f"{name}_model.joblib",
        model_dir / "reserve" / f"{name}_xgboost.json",
    ]
    if name == "prospectivity":
        candidates += [model_dir / "reserve_model.joblib", model_dir / "reserve_xgboost.json"]
    return next((path for path in candidates if path.exists()), None)


def load_booster_and_columns(model_path: Path, task: str = "classification"):
    columns = joblib.load(_feature_path(model_path))
    return _load_booster(str(model_path), task), columns


def load_model_bundle(model_path: Path, task: str = "classification"):
    """Load a trained model plus its feature schema.

    Accepts either artifact form written by the training pipeline:

    - ``{stem}_model.joblib`` — fitted estimator (any sklearn-compatible model,
      including XGBoost) persisted with joblib; pass the joblib path directly
    - ``{stem}.json`` — XGBoost-native booster; the joblib bundle is preferred
      when it exists alongside
    """
    columns = joblib.load(_feature_path(model_path))
    if model_path.suffix == ".joblib":
        return joblib.load(model_path), columns
    joblib_model_path = model_path.with_name(model_path.stem + "_model.joblib")
    if joblib_model_path.exists():
        return joblib.load(joblib_model_path), columns
    return _load_booster(str(model_path), task), columns


def load_model_metadata(model_path: Path) -> dict | None:
    """Return the training metadata sidecar for a model artifact, if present."""
    stem = model_path.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    meta_path = model_path.with_name(stem + "_meta.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def predict_with_model(df: pd.DataFrame, model_path: Path, task: str) -> pd.Series:
    prepared = ensure_spectral_indices(df)
    model, columns = load_model_bundle(model_path, task=task)
    features = prepare_reserve_matrix(prepared, columns)
    if task == "classification":
        return pd.Series(model.predict_proba(features)[:, 1], index=df.index)
    return pd.Series(model.predict(features), index=df.index)


def predict_prospectivity_frame(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    model_path = resolve_model_path(model_dir, "prospectivity")
    output = df.copy()
    if model_path is None:
        raise FileNotFoundError("No reserve prospectivity model artifact found")
    output["manganese_probability"] = predict_with_model(output, model_path, "classification")
    output["prospectivity_class"] = pd.cut(
        output["manganese_probability"],
        bins=[0, 0.25, 0.5, 0.75, 1],
        labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
        include_lowest=True,
    ).astype(str)
    return output


def maybe_predict_regressor(df: pd.DataFrame, model_path: Path) -> pd.Series | None:
    if model_path is None or not model_path.exists() or not _feature_path(model_path).exists():
        return None
    return predict_with_model(df, model_path, "regression")
