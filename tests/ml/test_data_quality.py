"""Phase 2 tests: data-quality pipeline calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services.data_quality import DATASET_CONTRACTS, DataQualityService


def test_data_quality_reports_meaningful_metrics_for_every_dataset(demo_store):
    report = DataQualityService(store=demo_store).run()
    names = {run["dataset_name"] for run in report["runs"]}
    assert names == set(DATASET_CONTRACTS)
    for run in report["runs"]:
        assert run["row_count"] > 0, run["dataset_name"]
        assert 0.0 <= run["missing_rate"] <= 1.0
        assert 0.0 <= run["duplicate_rate"] <= 1.0
        assert run["schema_valid"] is True
        assert 0.0 <= run["quality_score"] <= 1.0
    assert 0.0 <= report["overall_score"] <= 1.0


def test_quality_score_is_a_weighted_mean_of_check_flags(demo_store):
    report = DataQualityService(store=demo_store).run()
    for run in report["runs"]:
        checks = run["details"]["checks"]
        expected = sum(checks.values()) / len(checks)
        penalty = run["missing_rate"] * 0.5 + run["duplicate_rate"] * 0.25
        expected = max(0.0, min(1.0, expected - penalty))
        assert run["quality_score"] == pytest.approx(expected, abs=1e-3)


def test_contract_range_violations_are_reported(demo_store, tmp_path):
    service = DataQualityService(store=demo_store)
    contract = DATASET_CONTRACTS["production"]
    broken = pd.read_csv(demo_store.synthetic_dir / "production.csv").head(50)
    broken.loc[:, "production_mt"] = -100.0  # impossible production value
    path = tmp_path / "production.csv"
    broken.to_csv(path, index=False)

    result = service._check(path, "production", contract)
    assert result["details"]["contract_range_violations"] == ["production_mt"]
    assert result["details"]["checks"]["contract_ranges"] is False
    assert result["quality_score"] < 1.0


def test_missing_file_is_reported_as_invalid(tmp_path):
    service = DataQualityService()
    contract = DATASET_CONTRACTS["weather"]
    result = service._check(tmp_path / "does_not_exist.csv", "weather", contract)
    assert result["schema_valid"] is False
    assert result["quality_score"] == 0.0
    assert result["row_count"] == 0

