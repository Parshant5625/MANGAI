from __future__ import annotations

from pathlib import Path

from backend.app.core.config import Settings, get_settings


def _settings(settings: Settings | None) -> Settings:
    return settings or get_settings()


def _artifact_with_features_exists(artifact: Path, features: Path) -> bool:
    return artifact.is_file() and features.is_file()


def reserve_prospectivity_available(settings: Settings | None = None) -> bool:
    model_dir = _settings(settings).resolved_model_dir
    candidates = [
        (
            model_dir / "reserve" / "prospectivity_xgboost.json",
            model_dir / "reserve" / "prospectivity_xgboost_features.pkl",
        ),
        (model_dir / "reserve_xgboost.json", model_dir / "reserve_features.pkl"),
    ]
    return any(_artifact_with_features_exists(artifact, features) for artifact, features in candidates)


def production_forecast_available(settings: Settings | None = None) -> bool:
    model_dir = _settings(settings).resolved_model_dir
    return _artifact_with_features_exists(
        model_dir / "production" / "forecast_xgboost.json",
        model_dir / "production" / "forecast_features.pkl",
    )


def model_status(settings: Settings | None = None) -> dict[str, bool]:
    active_settings = _settings(settings)
    return {
        "reserve_prospectivity": reserve_prospectivity_available(active_settings),
        "production_forecast": production_forecast_available(active_settings),
    }
