from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_audit_event(audit_dir: Path, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Write a tamper-evident chained audit event without storing raw inputs."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "events.jsonl"
    previous_hash = ""
    if path.exists():
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    previous_hash = hashlib.sha256(line.rstrip(b"\n")).hexdigest()
    event = {
        "event_id": hashlib.sha256(f"{datetime.now(UTC).isoformat()}:{event_type}".encode()).hexdigest()[:24],
        "event_type": event_type,
        "occurred_at": datetime.now(UTC).isoformat(),
        "previous_event_hash": previous_hash,
        "payload": payload,
    }
    canonical = json.dumps(event, sort_keys=True, separators=(",", ":"))
    event["event_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def read_audit_events(audit_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    path = audit_dir / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events[-max(1, min(limit, 1000)):]
