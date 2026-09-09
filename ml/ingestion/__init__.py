"""Real-data ingestion skeleton for MANGAI (Phase 4).

This is a CLEAN INTERFACE, not a live connection. It does NOT connect to MOIL
systems, scrape external data, or modify the offline demo. It defines a
``DataSource`` protocol plus CSV/Parquet implementations and an
``IngestionPipeline`` that loads → validates (against Phase 2 contracts) →
normalizes → quality-checks → processes any tabular geological/satellite input.

Real field data can later be supplied as CSV / Parquet / a database source
implementing ``DataSource``; the demo system keeps working offline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """A tabular data source the ingestion pipeline can load."""

    name: str

    def load(self) -> pd.DataFrame:
        """Materialize the source as a DataFrame."""
        ...


class CsvSource:
    def __init__(self, path: str | Path, name: str | None = None, **read_kwargs: Any) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.read_kwargs = read_kwargs

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV source not found: {self.path}")
        return pd.read_csv(self.path, **self.read_kwargs)


class ParquetSource:
    def __init__(self, path: str | Path, name: str | None = None, **read_kwargs: Any) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.read_kwargs = read_kwargs

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet source not found: {self.path}")
        return pd.read_parquet(self.path, **self.read_kwargs)


@dataclass
class QualityReport:
    rows: int
    columns: int
    missingness: dict[str, float]
    duplicate_rows: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestionReport:
    source: str
    status: str
    quality: QualityReport
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    normalized_columns: list[str] = field(default_factory=list)


class IngestionPipeline:
    """load → validate (Phase 2 contracts) → normalize → quality-check → process.

    ``contract`` is an optional Phase 2 ``DatasetContract``; when provided the
    loaded frame is validated against it. The pipeline never mutates demo data.
    """

    def __init__(self, source: DataSource, *, contract: Any | None = None) -> None:
        self.source = source
        self.contract = contract

    def run(self) -> tuple[pd.DataFrame, IngestionReport]:
        frame = self.source.load()
        validation_errors: list[str] = []
        validation_warnings: list[str] = []
        if self.contract is not None:
            from ml.common.validation import validate_dataset

            result = validate_dataset(frame, self.contract)
            validation_errors = list(result.errors)
            validation_warnings = list(result.warnings)

        normalized = self._normalize(frame)
        quality = self._quality_check(normalized)
        status = "ok" if not validation_errors else "invalid"
        report = IngestionReport(
            source=self.source.name,
            status=status,
            quality=quality,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            normalized_columns=list(normalized.columns),
        )
        return normalized, report

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(column).strip() for column in normalized.columns]
        return normalized

    @staticmethod
    def _quality_check(frame: pd.DataFrame) -> QualityReport:
        missingness = {
            column: float(frame[column].isna().mean()) for column in frame.columns
        }
        duplicate_rows = int(frame.duplicated().sum())
        warnings = [
            f"column {column} is {rate:.1%} missing"
            for column, rate in missingness.items()
            if rate > 0.2
        ]
        return QualityReport(
            rows=int(len(frame)),
            columns=int(frame.shape[1]),
            missingness=missingness,
            duplicate_rows=duplicate_rows,
            warnings=warnings,
        )
