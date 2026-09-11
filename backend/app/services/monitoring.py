from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.demo_data import demo_envelope
from ml.common.monitoring import compare_feature_drift, registry_health


class ModelMonitoringService:
    """Read-only MLOps monitoring; it never changes model state or predictions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.registry_dir = self.settings.resolved_model_dir / "registry"

    def health(self) -> dict[str, Any]:
        registry = registry_health(self.registry_dir)
        drift_results = []
        alerts = list(registry.get("issues", []))

        # Registry records can optionally carry a reference/current feature
        # sample under `monitoring.baseline` and `monitoring.current`.
        # Existing records remain compatible and simply report no computed
        # drift until those populations are available.
        if self.registry_dir.exists():
            import json

            for path in sorted(self.registry_dir.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                monitoring = record.get("monitoring") or {}
                baseline = monitoring.get("baseline") or {}
                current = monitoring.get("current") or {}
                if baseline and current:
                    drift_results.extend(compare_feature_drift(baseline, current))

        for result in drift_results:
            if result.status in {"WARNING", "CRITICAL"}:
                alerts.append(
                    {
                        "type": "feature_drift",
                        "feature": result.feature,
                        "psi": result.psi,
                        "status": result.status,
                    }
                )

        status = "CRITICAL" if any(alert.get("status") == "CRITICAL" or alert.get("type") in {"missing_artifact", "invalid_registry_json", "multiple_champions", "champion_failed_leakage_check"} for alert in alerts) else "WARNING" if alerts else "HEALTHY"
        return {
            **demo_envelope(),
            "checked_at": registry["checked_at"],
            "registry": registry,
            "drift": [result.to_dict() for result in drift_results],
            "status": status,
            "alerts": alerts,
        }
