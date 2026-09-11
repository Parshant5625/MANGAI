from pathlib import Path

from ml.common.monitoring import classify_psi, compare_feature_drift, population_stability_index, registry_health


def test_psi_stable_for_same_distribution():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    assert population_stability_index(values, values) == 0.0
    assert classify_psi(0.02) == "STABLE"


def test_psi_detects_distribution_shift():
    baseline = [1, 2, 3, 4, 5, 6, 7, 8]
    current = [80, 90, 100, 110, 120, 130, 140, 150]
    result = population_stability_index(baseline, current)
    assert result > 0.25
    assert classify_psi(result) == "CRITICAL"


def test_feature_drift_is_deterministic():
    results = compare_feature_drift(
        {"rainfall": [1, 2, 3, 4], "downtime": [10, 11, 12, 13]},
        {"rainfall": [1, 2, 3, 4], "downtime": [100, 110, 120, 130]},
    )
    assert [item.feature for item in results] == ["downtime", "rainfall"]
    assert results[0].status == "CRITICAL"


def test_registry_health_reports_missing_registry(tmp_path: Path):
    result = registry_health(tmp_path / "registry")
    assert result["registry_exists"] is False
    assert result["status"] == "HEALTHY"
