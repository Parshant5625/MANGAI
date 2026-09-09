from __future__ import annotations

import pytest

from ml.common.external_contracts import (
    EXTERNAL_CONTRACTS,
    ExternalDataRequest,
    make_provenance,
    validate_external_batch,
)
from ml.common.provenance import DataBatch, DataProvenance


def test_external_contracts_cover_all_canonical_datasets() -> None:
    expected = {"geological", "boreholes", "satellite_features", "weather", "equipment", "blasting", "production"}
    assert set(EXTERNAL_CONTRACTS) == expected


def test_live_mode_rejects_synthetic_source() -> None:
    with pytest.raises(ValueError, match="live mode cannot use a synthetic source"):
        DataProvenance(
            source_name="demo",
            source_kind="synthetic",
            mode="live",
            dataset="weather",
        )


def test_batch_row_count_must_match_provenance() -> None:
    provenance = make_provenance(
        dataset="weather",
        source_name="demo",
        source_kind="synthetic",
        mode="demo",
        row_count=2,
    )
    with pytest.raises(ValueError, match="row_count"):
        DataBatch(records=[{"date": "2026-01-01"}], provenance=provenance)


def test_external_batch_validates_canonical_records() -> None:
    records = [
        {
            "date": "2026-01-01",
            "rainfall_mm": 5.0,
            "soil_moisture": 0.4,
            "temperature_c": 25.0,
            "vegetation_index": 0.5,
        }
    ]
    provenance = make_provenance(
        dataset="weather",
        source_name="demo_weather",
        source_kind="synthetic",
        mode="demo",
        row_count=1,
    )
    errors = validate_external_batch(DataBatch(records=records, provenance=provenance))
    assert errors == []


def test_external_batch_rejects_wrong_source_kind() -> None:
    records = [{
        "date": "2026-01-01",
        "rainfall_mm": 5.0,
        "soil_moisture": 0.4,
        "temperature_c": 25.0,
        "vegetation_index": 0.5,
    }]
    provenance = make_provenance(
        dataset="weather",
        source_name="satellite_feed",
        source_kind="satellite",
        mode="live",
        row_count=1,
    )
    errors = validate_external_batch(DataBatch(records=records, provenance=provenance))
    assert any("not accepted" in error for error in errors)


def test_external_request_rejects_invalid_date_window() -> None:
    with pytest.raises(ValueError, match="start must not be after end"):
        ExternalDataRequest(
            dataset="weather",
            site_id="demo-site",
            start="2026-02-01",
            end="2026-01-01",
        )
