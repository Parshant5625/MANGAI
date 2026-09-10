"""Canonical contracts for external-data adapters.

These contracts describe the provider-facing envelope. Individual providers
may use different APIs or file formats, but MANGAI receives the same metadata
and canonical record shape from every adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ml.common.contracts import ALL_CONTRACTS, DatasetContract
from ml.common.provenance import DataBatch, DataMode, DataProvenance, SourceKind

ExternalDataset = Literal[
    "geological",
    "boreholes",
    "satellite_features",
    "weather",
    "equipment",
    "blasting",
    "production",
]


@dataclass(frozen=True)
class ExternalDataRequest:
    """Provider-neutral request passed to an external adapter."""

    dataset: ExternalDataset
    site_id: str
    start: str
    end: str

    def __post_init__(self) -> None:
        if not self.site_id.strip():
            raise ValueError("site_id must not be empty")
        if not self.start.strip() or not self.end.strip():
            raise ValueError("start and end must not be empty")
        if self.start > self.end:
            raise ValueError("start must not be after end")


@dataclass(frozen=True)
class ExternalDataContract:
    """Maps an external dataset to its canonical MANGAI contract."""

    dataset: ExternalDataset
    canonical_contract: DatasetContract
    accepted_source_kinds: tuple[SourceKind, ...]
    required_provenance: tuple[str, ...] = (
        "source_name",
        "source_kind",
        "mode",
        "dataset",
        "ingested_at",
    )

    def validate_provenance(self, provenance: DataProvenance) -> list[str]:
        errors: list[str] = []
        if provenance.dataset != self.dataset:
            errors.append(
                f"Provenance dataset {provenance.dataset!r} does not match {self.dataset!r}."
            )
        if provenance.source_kind not in self.accepted_source_kinds:
            errors.append(
                f"Source kind {provenance.source_kind!r} is not accepted for {self.dataset!r}."
            )
        return errors


EXTERNAL_CONTRACTS: dict[str, ExternalDataContract] = {
    "geological": ExternalDataContract(
        "geological", ALL_CONTRACTS["geological"], ("synthetic", "local_file", "database")
    ),
    "boreholes": ExternalDataContract(
        "boreholes", ALL_CONTRACTS["boreholes"], ("synthetic", "local_file", "database")
    ),
    "satellite_features": ExternalDataContract(
        "satellite_features", ALL_CONTRACTS["satellite_features"], ("synthetic", "local_file", "satellite", "api")
    ),
    "weather": ExternalDataContract(
        "weather", ALL_CONTRACTS["weather"], ("synthetic", "local_file", "weather", "api")
    ),
    "equipment": ExternalDataContract(
        "equipment", ALL_CONTRACTS["equipment"], ("synthetic", "local_file", "sensor", "database")
    ),
    "blasting": ExternalDataContract(
        "blasting", ALL_CONTRACTS["blasting"], ("synthetic", "local_file", "database")
    ),
    "production": ExternalDataContract(
        "production", ALL_CONTRACTS["production"], ("synthetic", "local_file", "database")
    ),
}


def validate_external_batch(batch: DataBatch) -> list[str]:
    """Validate provenance and canonical records for an external batch."""
    contract = EXTERNAL_CONTRACTS.get(batch.provenance.dataset)
    if contract is None:
        return [f"Unknown external dataset: {batch.provenance.dataset!r}"]

    errors = contract.validate_provenance(batch.provenance)
    if batch.provenance.mode == "live" and batch.provenance.is_synthetic:
        errors.append("Live ingestion cannot be marked synthetic.")

    # Import lazily to keep this module usable by lightweight provider tooling.
    import pandas as pd

    from ml.common.validation import validate_dataset

    validation = validate_dataset(pd.DataFrame(batch.records), contract.canonical_contract)
    errors.extend(validation.errors)
    return errors


def canonical_contract(dataset: ExternalDataset) -> DatasetContract:
    """Return the canonical tabular contract for an external dataset."""
    return EXTERNAL_CONTRACTS[dataset].canonical_contract


def make_provenance(
    *,
    dataset: ExternalDataset,
    source_name: str,
    source_kind: SourceKind,
    mode: DataMode,
    row_count: int,
    **kwargs: Any,
) -> DataProvenance:
    """Construct provider provenance with consistent dataset and row metadata."""
    return DataProvenance(
        source_name=source_name,
        source_kind=source_kind,
        mode=mode,
        dataset=dataset,
        row_count=row_count,
        **kwargs,
    )
