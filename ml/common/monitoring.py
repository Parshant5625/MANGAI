from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DriftResult:
    feature: str
    psi: float
    status: str
    baseline_count: int
    current_count: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "status": self.status,
            "baseline_count": self.baseline_count,
            "current_count": self.current_count,
            "note": self.note,
        }


def _finite_numeric(values: Iterable[Any]) -> np.ndarray:
    numeric: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            numeric.append(parsed)
    return np.asarray(numeric, dtype=np.float64)


def population_stability_index(
    baseline: Iterable[Any],
    current: Iterable[Any],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI without mutating either population.

    Quantile bins are derived from the baseline population so the comparison
    remains anchored to the training/reference distribution.
    """
    reference = _finite_numeric(baseline)
    observed = _finite_numeric(current)
    if reference.size < 2 or observed.size < 2:
        return 0.0
    bins = max(2, min(int(bins), 20))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(observed, bins=edges)
    ref_pct = np.maximum(ref_counts / reference.size, epsilon)
    cur_pct = np.maximum(cur_counts / observed.size, epsilon)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi: float, *, warning: float = 0.10, critical: float = 0.25) -> str:
    if psi >= critical:
        return "CRITICAL"
    if psi >= warning:
        return "WARNING"
    return "STABLE"


def compare_feature_drift(
    baseline: dict[str, Iterable[Any]],
    current: dict[str, Iterable[Any]],
    *,
    warning: float = 0.10,
    critical: float = 0.25,
) -> list[DriftResult]:
    results: list[DriftResult] = []
    for feature in sorted(set(baseline) & set(current)):
        base_values = _finite_numeric(baseline[feature])
        current_values = _finite_numeric(current[feature])
        psi = population_stability_index(base_values, current_values)
        results.append(
            DriftResult(
                feature=feature,
                psi=psi,
                status=classify_psi(psi, warning=warning, critical=critical),
                baseline_count=int(base_values.size),
                current_count=int(current_values.size),
                note="Insufficient numeric observations; PSI reported as 0." if min(base_values.size, current_values.size) < 2 else "",
            )
        )
    return results


def registry_health(registry_dir: Path) -> dict[str, Any]:
    """Audit registry records and artifact integrity without changing them."""
    now = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if registry_dir.exists():
        for path in sorted(registry_dir.glob("*.json")):
            try:
                import json

                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append({"type": "invalid_registry_json", "path": str(path), "reason": str(exc)})
                continue
            records.append(record)
            artifact = record.get("artifact_path")
            if artifact:
                artifact_path = Path(artifact)
                if not artifact_path.is_absolute():
                    artifact_path = registry_dir.parent.parent / artifact
                if not artifact_path.exists():
                    issues.append({"type": "missing_artifact", "model_name": record.get("model_name"), "version": record.get("version"), "artifact_path": str(artifact_path)})
            if record.get("status") == "champion" and record.get("leakage_check_passed") is False:
                issues.append({"type": "champion_failed_leakage_check", "model_name": record.get("model_name"), "version": record.get("version")})
    champions: dict[str, str] = {}
    for record in records:
        if record.get("status") == "champion":
            name = str(record.get("model_name"))
            version = str(record.get("version"))
            if name in champions:
                issues.append({"type": "multiple_champions", "model_name": name, "versions": [champions[name], version]})
            champions[name] = version
    return {
        "checked_at": now.isoformat(),
        "registry_exists": registry_dir.exists(),
        "record_count": len(records),
        "champions": champions,
        "issue_count": len(issues),
        "issues": issues,
        "status": "CRITICAL" if issues else "HEALTHY",
    }
