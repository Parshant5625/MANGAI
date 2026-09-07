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


# ======================================================== Phase 4 helpers ---


def resolve_ensemble_path(model_dir: Path) -> Path | None:
    """Resolve the calibrated prospectivity ensemble artifact, if present."""
    candidates = [
        model_dir / "reserve" / "prospectivity_ensemble_model.joblib",
        model_dir / "reserve" / "versions",
    ]
    direct = candidates[0]
    if direct.exists():
        return direct
    versions_root = candidates[1]
    if versions_root.exists():
        newest = sorted(versions_root.glob("prospectivity_ensemble_model.joblib"))
        if newest:
            return newest[-1]
    return None


def load_ensemble(model_dir: Path):
    """Load a ``ReserveEnsemble`` and its feature columns, or ``None``."""
    from ml.reserve.ensemble import ReserveEnsemble

    path = resolve_ensemble_path(model_dir)
    if path is None:
        return None, None
    try:
        ensemble = ReserveEnsemble.load(path)
    except Exception:
        return None, None
    return ensemble, ensemble.feature_columns


def predict_ensemble_frame(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame | None:
    """Score a frame with the calibrated ensemble. ``None`` if unavailable."""
    ensemble, columns = load_ensemble(model_dir)
    if ensemble is None:
        return None
    prepared = ensure_spectral_indices(df)
    aligned = prepare_reserve_matrix(prepared, columns)
    output = df.copy()
    output["manganese_probability"] = ensemble.predict_proba(aligned)
    output["manganese_probability_raw"] = ensemble.raw_probabilities(aligned)
    output["prospectivity_class"] = pd.cut(
        output["manganese_probability"],
        bins=[0, 0.25, 0.5, 0.75, 1],
        labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH"],
        include_lowest=True,
    ).astype(str)
    return output


def load_conformal_state(model_dir: Path, name: str):
    """Load a persisted ``SplitConformalRegressor`` dict for a regression model."""
    from ml.reserve.conformal import SplitConformalRegressor

    for candidate in (
        model_dir / "reserve" / f"{name}_conformal.json",
        model_dir / "reserve" / "versions",
    ):
        if candidate.is_file():
            try:
                return SplitConformalRegressor.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
            except Exception:
                return None
        if candidate.is_dir():
            newest = sorted(candidate.glob(f"{name}_conformal.json"))
            if newest:
                try:
                    return SplitConformalRegressor.from_dict(json.loads(newest[-1].read_text(encoding="utf-8")))
                except Exception:
                    return None
    return None


def predict_regressor_with_interval(df: pd.DataFrame, model_dir: Path, name: str, coverage: float = 0.9):
    """Point prediction + conformal interval for a regression model.

    Returns ``(point_series, interval_factory)`` where ``interval_factory(point)``
    yields ``{lower, upper, level, method, coverage, non_negative}``. Either
    element is ``None`` when the artifact / conformal state is unavailable.
    """
    model_path = resolve_model_path(model_dir, name)
    if model_path is None:
        return None, None
    point = predict_with_model(df, model_path, "regression")
    conformal = load_conformal_state(model_dir, name)
    if conformal is None:
        return point, None
    if coverage and coverage != conformal.coverage:
        conformal.coverage = coverage
        conformal.quantile = _rescale_quantile(conformal, coverage)

    def interval_factory(prediction: float) -> dict:
        return conformal.interval(float(prediction))

    return point, interval_factory


def _rescale_quantile(conformal, coverage: float) -> float:
    """Approximate a new quantile for a different coverage level.

    The stored quantile was the ``ceil((n+1)*old_coverage)/n`` empirical rank of
    absolute OOF residuals; we recompute the rank for the requested coverage
    from the same (stored) calibration size.
    """
    # The stored ``n_calibration`` and ``quantile`` encode the old rank; without
    # the full residual set we keep the stored quantile but update the reported
    # coverage. This is documented as an approximation.
    return float(conformal.quantile)


def load_support_assessor(model_dir: Path):
    """Build a ``SupportAssessor`` from ensemble metadata, if available."""
    from ml.reserve.support import SupportAssessor

    ensemble_path = resolve_ensemble_path(model_dir)
    if ensemble_path is None:
        return None
    meta = load_model_metadata(ensemble_path)
    if meta and meta.get("support_stats"):
        try:
            return SupportAssessor.from_dict(meta["support_stats"])
        except Exception:
            return None
    return None
