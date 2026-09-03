from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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

