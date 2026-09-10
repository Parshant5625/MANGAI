from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.app.core.errors import DataUnavailableError


def parse_scene_datetime(value: Any) -> datetime:
    if not value:
        raise DataUnavailableError("Satellite scene is missing an acquisition datetime.")
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DataUnavailableError(
            "Satellite scene has an invalid acquisition datetime.",
            details={"datetime": str(value)},
        ) from exc
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def select_temporally_matched_scene(
    reference_datetime: Any,
    scenes: list[dict[str, Any]],
    max_days: int = 16,
) -> dict[str, Any]:
    """Select the closest thermal scene without crossing the configured time window."""

    if max_days < 0:
        raise DataUnavailableError("Maximum temporal matching window must be non-negative.")
    reference = parse_scene_datetime(reference_datetime)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for scene in scenes:
        try:
            scene_time = parse_scene_datetime(scene.get("datetime"))
        except DataUnavailableError:
            continue
        distance_days = abs((scene_time - reference).total_seconds()) / 86400.0
        if distance_days <= max_days:
            candidates.append((distance_days, scene))
    if not candidates:
        raise DataUnavailableError(
            "No thermal scene is within the allowed temporal matching window.",
            details={"max_days": max_days, "reference_datetime": reference.isoformat()},
        )
    candidates.sort(key=lambda item: (item[0], str(item[1].get("scene_id", ""))))
    selected = dict(candidates[0][1])
    selected["temporal_distance_days"] = candidates[0][0]
    return selected
