from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope
from backend.app.services.model_artifacts import reserve_prospectivity_available

# Directory kept for import-time clarity; registry records live here.
FALLBACK_NOTE = "Synthetic-data candidate. Not field-validated."


class ModelRegistryService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def list_models(self) -> dict[str, Any]:
        registry_dir = self.settings.resolved_model_dir / "registry"
        models: list[dict[str, Any]] = []
        if registry_dir.exists():
            from ml.common.registry import compare_models

            for record in compare_models(registry_dir):
                record.setdefault("notes", FALLBACK_NOTE)
                record.setdefault(
                    "drift",
                    {
                        "status": "unknown_demo",
                        "note": "No MOIL production baseline. Drift monitoring is a placeholder until field data is connected.",
                    },
                )
                models.append(record)
        if not models:
            models = self._fallback_catalog()
        return {**demo_envelope(), "models": models}

    def promote(self, model_name: str, version: str, to_status: str) -> dict[str, Any]:
        from ml.common.registry import promote_model

        registry_dir = self.settings.resolved_model_dir / "registry"
        return promote_model(registry_dir, model_name, version, to_status)

    def compare(self, model_name: str | None = None) -> dict[str, Any]:
        from ml.common.registry import compare_models

        registry_dir = self.settings.resolved_model_dir / "registry"
        return {**demo_envelope(), "comparison": compare_models(registry_dir, model_name=model_name)}

    def _fallback_catalog(self) -> list[dict[str, Any]]:
        metrics_path = self.settings.resolved_model_dir / "reserve_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"status": "missing"}
        artifact_path = self.settings.resolved_model_dir / "reserve_xgboost.json"
        created_at = (
            datetime.fromtimestamp(artifact_path.stat().st_mtime, UTC).isoformat()
            if artifact_path.exists()
            else datetime.fromtimestamp(0, UTC).isoformat()
        )
        available = reserve_prospectivity_available(self.settings)
        return [
            {
                "model_name": "reserve_prospectivity",
                "version": "reserve-xgb-001",
                "task": "binary_classification",
                "algorithm": "XGBoost",
                "training_data_hash": "unhashed",
                "feature_schema_hash": "unhashed",
                "metrics": metrics,
                "artifact_path": "models/reserve_xgboost.json",
                "created_at": created_at,
                "status": "available" if available else "missing",
                "notes": "Synthetic-data model; not field-validated." if available else "Model artifact is not available.",
            }
        ]
