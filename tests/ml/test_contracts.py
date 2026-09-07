"""Phase 2 tests: canonical data contracts and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.common.contracts import (
    ALL_CONTRACTS,
    BLASTING_CONTRACT,
    BOREHOLE_CONTRACT,
    EQUIPMENT_CONTRACT,
    GEOLOGICAL_CONTRACT,
    PRODUCTION_CONTRACT,
    SATELLITE_CONTRACT,
    WEATHER_CONTRACT,
)
from ml.common.validation import assert_valid, validate_borehole_intervals, validate_dataset

# ============================================================
# A. Schema validation
# ============================================================


def test_all_eight_dataset_contracts_are_registered():
    expected = {
        "geological",
        "boreholes",
        "satellite_features",
        "weather",
        "equipment",
        "equipment_events",
        "blasting",
        "production",
    }
    assert set(ALL_CONTRACTS) == expected


def test_every_numeric_contract_column_documents_its_unit():
    for contract in ALL_CONTRACTS.values():
        for column in contract.columns:
            if column.dtype in ("float", "int"):
                assert column.unit, f"{contract.name}.{column.name} missing unit"


def test_geological_contract_defines_coordinate_and_grade_units():
    assert GEOLOGICAL_CONTRACT.column("latitude").unit == "decimal degrees"
    assert GEOLOGICAL_CONTRACT.column("mn_pct").unit == "percent"
    assert GEOLOGICAL_CONTRACT.column("ore_thickness_m").unit == "meters"


def test_demo_csv_files_satisfy_their_contracts(demo_store):
    assert_valid(demo_store.geological(), GEOLOGICAL_CONTRACT)
    assert_valid(demo_store.satellite(), SATELLITE_CONTRACT)
    assert_valid(demo_store.boreholes(), BOREHOLE_CONTRACT)
    assert_valid(demo_store.weather(), WEATHER_CONTRACT)
    assert_valid(demo_store.equipment(), EQUIPMENT_CONTRACT)
    assert_valid(demo_store.blasting(), BLASTING_CONTRACT)
    assert_valid(demo_store.production(), PRODUCTION_CONTRACT)


# ============================================================
# B. Invalid data rejection
# ============================================================


def _minimal_geological_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["GS00001"],
            "latitude": [21.4],
            "longitude": [80.3],
            "elevation_m": [650.0],
            "slope_deg": [10.0],
            "aspect_deg": [180.0],
            "depth_m": [30.0],
            "formation": ["Gondite"],
            "mn_pct": [12.0],
            "fe_pct": [15.0],
            "sio2_pct": [35.0],
            "ore_thickness_m": [5.0],
            "is_manganese": [1],
        }
    )


def test_out_of_range_latitude_is_rejected():
    df = _minimal_geological_frame()
    df.loc[0, "latitude"] = 95.0
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert not result.valid
    assert any("latitude" in error for error in result.errors)


def test_impossible_grade_is_rejected():
    df = _minimal_geological_frame()
    df.loc[0, "mn_pct"] = 120.0
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert not result.valid
    assert any("mn_pct" in error for error in result.errors)


def test_invalid_formation_category_is_rejected():
    df = _minimal_geological_frame()
    df.loc[0, "formation"] = "Unobtanium"
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert not result.valid
    assert any("formation" in error for error in result.errors)


def test_duplicate_primary_key_is_rejected():
    df = pd.concat([_minimal_geological_frame(), _minimal_geological_frame()], ignore_index=True)
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert not result.valid
    assert any("sample_id" in error for error in result.errors)


def test_missing_required_column_is_rejected():
    df = _minimal_geological_frame().drop(columns=["depth_m"])
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert not result.valid
    assert any("depth_m" in error for error in result.errors)


def test_extra_column_is_a_warning_not_an_error():
    df = _minimal_geological_frame()
    df["unexpected"] = 1.0
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert result.valid
    assert any("unexpected" in warning for warning in result.warnings)


def test_empty_dataset_is_rejected():
    result = validate_dataset(_minimal_geological_frame().iloc[0:0], GEOLOGICAL_CONTRACT)
    assert not result.valid


def test_null_in_non_nullable_column_is_rejected():
    df = _minimal_geological_frame()
    df.loc[0, "latitude"] = None
    result = validate_dataset(df, GEOLOGICAL_CONTRACT)
    assert not result.valid


def test_assert_valid_raises_value_error():
    df = _minimal_geological_frame()
    df.loc[0, "slope_deg"] = -5.0
    with pytest.raises(ValueError):
        assert_valid(df, GEOLOGICAL_CONTRACT)


def test_borehole_inverted_intervals_are_rejected():
    df = pd.DataFrame({"from_depth_m": [10.0, 50.0], "to_depth_m": [20.0, 40.0]})
    result = validate_borehole_intervals(df)
    assert not result.valid
    assert any("to_depth_m" in error for error in result.errors)


def test_borehole_valid_intervals_pass():
    df = pd.DataFrame({"from_depth_m": [10.0, 20.0], "to_depth_m": [20.0, 40.0]})
    assert validate_borehole_intervals(df).valid
