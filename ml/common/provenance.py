"""Provenance contracts for external and local MANGAI data.

The ingestion layer keeps the canonical tabular records separate from their
provenance. This makes live/demo origin explicit and gives downstream code a
stable way to reason about freshness, quality, and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

DataMode = Literal["demo", "live"]
SourceKind = Literal["synthetic", "local_file", "api", "satellite", "weather", "sensor", "database"]


@dataclass(frozen=True)
class DataProvenance:
    """Source metadata attached to one ingested dataset/batch."""

    source_name: str
    source_kind: SourceKind
    mode: DataMode
    dataset: str
    acquired_at: str | None = None
    ingested_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_version: str | None = None
    source_uri: str | None = None
    checksum: str | None = None
    license_note: str | None = None
    quality_score: float | None = None
    row_count: int | None = None

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.dataset.strip():
            raise ValueError("dataset must not be empty")
        if self.mode == "live" and self.source_kind == "synthetic":
            raise ValueError("live mode cannot use a synthetic source")
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0 and 1")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("row_count cannot be negative")

    @property
    def is_synthetic(self) -> bool:
        return self.source_kind == "synthetic"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_kind": self.source_kind,
            "mode": self.mode,
            "dataset": self.dataset,
            "acquired_at": self.acquired_at,
            "ingested_at": self.ingested_at,
            "source_version": self.source_version,
            "source_uri": self.source_uri,
            "checksum": self.checksum,
            "license_note": self.license_note,
            "quality_score": self.quality_score,
            "row_count": self.row_count,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class DataBatch:
    """Canonical records plus provenance for an ingestion operation."""

    records: list[dict[str, Any]]
    provenance: DataProvenance

    def __post_init__(self) -> None:
        if self.provenance.row_count is not None and self.provenance.row_count != len(self.records):
            raise ValueError("provenance row_count does not match records")

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "provenance": self.provenance.to_dict(),
        }


def provenance_from_records(
    records: list[dict[str, Any]],
    *,
    source_name: str,
    source_kind: SourceKind,
    mode: DataMode,
    dataset: str,
    **kwargs: Any,
) -> DataBatch:
    """Create a provenance-aware batch and stamp its row count."""
    provenance = DataProvenance(
        source_name=source_name,
        source_kind=source_kind,
        mode=mode,
        dataset=dataset,
        row_count=len(records),
        **kwargs,
    )
    return DataBatch(records=records, provenance=provenance)
