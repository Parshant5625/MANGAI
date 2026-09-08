from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelVersionRecord:
    model_name: str
    version: str
    task: str
    algorithm: str
    training_data_hash: str
    feature_schema_hash: str
    metrics: dict[str, Any]
    artifact_path: str
    status: str = "candidate"
    created_at: str = ""
    target: str = ""
    feature_names: list[str] = field(default_factory=list)
    validation_strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = payload["created_at"] or datetime.now(UTC).isoformat()
        return payload


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_schema(schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def write_registry_record(record: ModelVersionRecord, registry_dir: Path) -> Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{record.model_name}-{record.version}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2)
    return path


# ============================================================ lifecycle ---

LIFECYCLE_STATUSES = ("candidate", "validated", "champion", "retired")
PROMOTION_GRAPH = {
    "candidate": ("validated",),
    "validated": ("champion",),
    "champion": ("retired",),
    "retired": (),
}


def _read_record(registry_dir: Path, model_name: str, version: str) -> dict[str, Any] | None:
    path = registry_dir / f"{model_name}-{version}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_record(registry_dir: Path, record: dict[str, Any]) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{record['model_name']}-{record['version']}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


def check_promotion_criteria(
    record: dict[str, Any],
    *,
    required_metrics: tuple[str, ...] = ("validation", "leakage_check_passed"),
) -> tuple[bool, list[str]]:
    """Deterministic, documented criteria for promotion.

    A model may advance only when ALL hold:
    - ``leakage_check_passed`` is true
    - ``validation`` starts with ``spatial_block_holdout``
    - every ``required_metric`` is present and not None
    - the artifact file exists on disk (if ``artifact_path`` is set)
    """
    failures: list[str] = []
    metrics_payload = record.get("metrics", {})
    if not (record.get("leakage_check_passed") or metrics_payload.get("leakage_check_passed")):
        failures.append("leakage_check_passed is not true")
    validation = record.get("validation") or metrics_payload.get("validation", "")
    if not str(validation).startswith("spatial_block_holdout"):
        failures.append(f"validation strategy {validation!r} is not spatial_block_holdout")
    for metric in required_metrics:
        if metrics_payload.get(metric) is None and record.get(metric) is None:
            failures.append(f"required metric {metric!r} is missing")
    artifact_path = record.get("artifact_path")
    if artifact_path and not Path(artifact_path).exists():
        failures.append(f"artifact not found: {artifact_path}")
    return (not failures, failures)


def promote_model(
    registry_dir: Path,
    model_name: str,
    version: str,
    to_status: str,
    *,
    required_metrics: tuple[str, ...] = ("validation", "leakage_check_passed"),
    demote_previous_champion: bool = True,
) -> dict[str, Any]:
    """Deterministic, auditable promotion along candidate→validated→champion.

    Promotion requires the criteria in ``check_promotion_criteria`` to pass.
    When promoting to ``champion``, the existing champion (if any) is demoted
    to ``validated`` and recorded as ``previous_champion``. Old versions and
    artifacts are never overwritten.
    """
    if to_status not in LIFECYCLE_STATUSES:
        raise ValueError(f"unknown status {to_status!r}; expected one of {LIFECYCLE_STATUSES}")
    record = _read_record(registry_dir, model_name, version)
    if record is None:
        raise FileNotFoundError(f"no registry record for {model_name} {version}")
    current = record.get("status", "candidate")
    allowed = PROMOTION_GRAPH.get(current, ())
    if to_status not in allowed:
        raise ValueError(f"illegal promotion {current} -> {to_status}; allowed: {allowed}")

    if to_status in ("validated", "champion"):
        passed, failures = check_promotion_criteria(record, required_metrics=required_metrics)
        if not passed:
            raise ValueError(f"promotion criteria not met: {'; '.join(failures)}")

    promotion_history = list(record.get("promotion_history", []))
    now = datetime.now(UTC).isoformat()
    previous_champion = None
    if to_status == "champion" and demote_previous_champion:
        for path in sorted(registry_dir.glob(f"{model_name}-*.json")):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate.get("status") == "champion" and candidate.get("version") != version:
                previous_champion = candidate.get("version")
                candidate["status"] = "validated"
                candidate["demoted_at"] = now
                candidate["demotion_reason"] = f"superseded by {version}"
                _write_record(registry_dir, candidate)

    record["status"] = to_status
    record["promoted_at"] = now
    if to_status == "champion":
        record["previous_champion"] = previous_champion
    promotion_history.append({"from": current, "to": to_status, "at": now})
    record["promotion_history"] = promotion_history
    _write_record(registry_dir, record)
    return record


def load_registry_records(registry_dir: Path, *, model_name: str | None = None) -> list[dict[str, Any]]:
    """Load registry records, optionally filtered by model name."""
    if not registry_dir.exists():
        return []
    records = []
    for path in sorted(registry_dir.glob(f"{model_name or '*'}-*.json")):
        if model_name is None or path.name.startswith(f"{model_name}-"):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def compare_models(registry_dir: Path, *, model_name: str | None = None) -> list[dict[str, Any]]:
    """Reusable model-comparison view across registry records.

    ``created_at`` is taken from the record when present; legacy records
    written before the field existed fall back to the registry file's own
    modification time (real filesystem provenance, never a fabricated stamp).
    """
    comparison = []
    if not registry_dir.exists():
        return []
    for path in sorted(registry_dir.glob(f"{model_name or '*'}-*.json")):
        if model_name is None or path.name.startswith(f"{model_name}-"):
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            continue
        metrics = record.get("metrics", {})
        created_at = record.get("created_at") or datetime.fromtimestamp(
            path.stat().st_mtime, UTC
        ).isoformat()
        comparison.append(
            {
                "model_name": record.get("model_name"),
                "version": record.get("version"),
                "task": record.get("task"),
                "algorithm": record.get("algorithm"),
                "status": record.get("status"),
                "validation_strategy": record.get("validation") or metrics.get("validation"),
                "metrics": metrics,
                "feature_schema_hash": record.get("feature_schema_hash"),
                "training_data_hash": record.get("training_data_hash"),
                "leakage_check_passed": metrics.get("leakage_check_passed"),
                "artifact_path": record.get("artifact_path"),
                "artifact_exists": Path(record["artifact_path"]).exists() if record.get("artifact_path") else None,
                "promoted_at": record.get("promoted_at"),
                "previous_champion": record.get("previous_champion"),
                "created_at": created_at,
            }
        )
    return comparison


