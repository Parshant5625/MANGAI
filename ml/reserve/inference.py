from __future__ import annotations

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
    if model_path.name == "reserve_xgboost.json":
        return model_path.with_name("reserve_features.pkl")
    return model_path.with_name(model_path.stem + "_features.pkl")


def load_booster_and_columns(model_path: Path, task: str = "classification"):
    columns = joblib.load(_feature_path(model_path))
    return _load_booster(str(model_path), task), columns


def predict_with_model(df: pd.DataFrame, model_path: Path, task: str) -> pd.Series:
    prepared = ensure_spectral_indices(df)
    model, columns = load_booster_and_columns(model_path, task=task)
    features = prepare_reserve_matrix(prepared, columns)
    if task == "classification":
        return pd.Series(model.predict_proba(features)[:, 1], index=df.index)
    return pd.Series(model.predict(features), index=df.index)


def predict_prospectivity_frame(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    candidates = [model_dir / "reserve" / "prospectivity_xgboost.json", model_dir / "reserve_xgboost.json"]
    model_path = next((path for path in candidates if path.exists()), None)
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
    if not model_path.exists() or not _feature_path(model_path).exists():
        return None
    return predict_with_model(df, model_path, "regression")
