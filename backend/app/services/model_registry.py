from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope


class ModelRegistryService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def list_models(self) -> dict[str, Any]:
        registry_dir = self.settings.resolved_model_dir / "registry"
        models: list[dict[str, Any]] = []
        if registry_dir.exists():
            for path in sorted(registry_dir.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.setdefault("notes", "Synthetic-data candidate. Not field-validated.")
                payload.setdefault("feature_schema_hash", payload.get("feature_schema_hash", ""))
                payload.setdefault(
                    "drift",
                    {
                        "status": "unknown_demo",
                        "note": "No MOIL production baseline. Drift monitoring is a placeholder until field data is connected.",
                    },
                )
                models.append(payload)
        if not models:
            models = self._fallback_catalog()
        return {**demo_envelope(), "models": models}

    def _fallback_catalog(self) -> list[dict[str, Any]]:
        metrics_path = self.settings.resolved_model_dir / "reserve_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"status": "missing"}
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
                "created_at": "",
                "status": "candidate",
                "notes": "Synthetic-data model; not field-validated.",
            }
        ]
