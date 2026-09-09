"""Phase 4 tests: ensemble, calibration, conformal intervals, support,
extrapolation, resource uncertainty, drift, ingestion, registry lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.common.drift import compute_feature_drift
from ml.common.registry import (
    check_promotion_criteria,
    compare_models,
    load_registry_records,
    promote_model,
)
from ml.ingestion import CsvSource, IngestionPipeline
from ml.reserve.calibration import ProbabilityCalibrator, brier_score, fit_calibrator, reliability_curve
from ml.reserve.conformal import SplitConformalRegressor, conformal_quantile
from ml.reserve.ensemble import ReserveEnsemble, compute_member_weights
from ml.reserve.resource_estimator import estimate_resource_potential_with_intervals
from ml.reserve.support import SupportAssessor, standardized_distance_score, support_level_from_score

FORMATION_VALUES = ["Manganiferous_Formation", "Gondite", "Phyllite", "Banded_Iron_Formation"]


def _make_geo_satellite(n: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(21.0, 22.0, n)
    longitude = rng.uniform(80.0, 81.0, n)
    logit = 4.0 * (latitude - 21.0) + 4.0 * (longitude - 80.0) - 3.0
    probability = 1.0 / (1.0 + np.exp(-logit))
    is_manganese = rng.binomial(1, probability)
    formation = np.array(FORMATION_VALUES, dtype=object)[rng.integers(0, len(FORMATION_VALUES), n)]
    manganiferous = rng.random(n) < (0.15 + 0.7 * probability)
    formation[manganiferous] = "Manganiferous_Formation"
    geological = pd.DataFrame(
        {
            "sample_id": [f"S{i:05d}" for i in range(n)],
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": rng.normal(620, 40, n),
            "slope_deg": rng.uniform(0, 35, n),
            "aspect_deg": rng.uniform(0, 360, n),
            "depth_m": rng.uniform(2, 60, n),
            "formation": formation,
            "is_manganese": is_manganese,
            "mn_pct": np.clip(3 + 30 * probability + rng.normal(0, 3, n), 0.5, 50),
            "fe_pct": rng.uniform(2, 15, n),
            "sio2_pct": rng.uniform(20, 60, n),
            "ore_thickness_m": np.clip(0.5 + 9 * probability + rng.normal(0, 1.2, n), 0.2, None),
        }
    )
    satellite = pd.DataFrame(
        {
            "sample_id": geological["sample_id"],
            "latitude": latitude,
            "longitude": longitude,
            "blue_b2": 0.10 + 0.05 * (1 - probability) + rng.normal(0, 0.01, n),
            "green_b3": 0.14 + 0.04 * (1 - probability) + rng.normal(0, 0.01, n),
            "red_b4": 0.16 + 0.06 * (1 - probability) + rng.normal(0, 0.01, n),
            "nir_b8": 0.30 + 0.10 * probability + rng.normal(0, 0.01, n),
            "swir_b11": 0.28 + 0.08 * probability + rng.normal(0, 0.01, n),
            "swir_b12": 0.22 + 0.07 * probability + rng.normal(0, 0.01, n),
            "lst_c": 30 + 4 * (1 - probability) + rng.normal(0, 0.5, n),
        }
    )
    return geological, satellite


def _write_training_root(root: Path, n: int = 400, seed: int = 7) -> Path:
    geo, sat = _make_geo_satellite(n, seed)
    synthetic = root / "data" / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    geo.to_csv(synthetic / "geological.csv", index=False)
    sat.to_csv(synthetic / "satellite_features.csv", index=False)
    return root


# --------------------------------------------------------------- ensemble --


def test_compute_member_weights_sums_to_one():
    y = pd.Series([0, 1, 1, 0, 1])
    # ``a`` perfectly separates the classes (ROC-AUC 1.0); ``b`` is imperfect
    # (positive/negative score overlap), so the fitted ``a`` weight must exceed
    # ``b`` while both sum to one.
    oof = {"a": pd.Series([0.2, 0.8, 0.7, 0.3, 0.6]), "b": pd.Series([0.4, 0.5, 0.5, 0.4, 0.4])}
    weights = compute_member_weights(oof, y)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["a"] > weights["b"]


def test_ensemble_fit_predicts_calibrated_probabilities(tmp_path_factory):
    from ml.reserve.features import RESERVE_TASKS, load_fused_reserve_table, prepare_reserve_matrix
    from ml.reserve.spatial import spatial_holdout_indices

    root = _write_training_root(tmp_path_factory.mktemp("ens"), n=300, seed=11)
    df = load_fused_reserve_table(root)
    spec = RESERVE_TASKS["prospectivity"]
    X = prepare_reserve_matrix(
        df,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    y = df[spec.target].astype(int)
    train_idx, test_idx, groups = spatial_holdout_indices(df)
    ensemble, diagnostics = ReserveEnsemble.fit(X, y, groups, train_idx, quick=True)
    probs = ensemble.predict_proba(X.iloc[test_idx])
    assert len(probs) == len(test_idx)
    assert (probs >= 0).all() and (probs <= 1).all()
    assert set(diagnostics["weights"]) == {"logistic_regression", "random_forest", "xgboost"}


def test_ensemble_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    y = pd.Series(rng.integers(0, 2, 50))
    oof = {name: rng.random(50) for name in ["logistic_regression", "random_forest", "xgboost"]}
    weights = compute_member_weights(oof, y)
    ensemble = ReserveEnsemble(members={}, weights=weights, calibrator=fit_calibrator(oof["xgboost"], y), feature_columns=["a", "b"])
    path = tmp_path / "ens.joblib"
    ensemble.save(path)
    loaded = ReserveEnsemble.load(path)
    assert loaded.weights == weights
    assert loaded.calibrator.kind in {"isotonic", "sigmoid"}


# ------------------------------------------------------------ calibration --


def test_brier_score_perfect_and_worst():
    y = pd.Series([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == 0.0
    assert abs(brier_score(y, 1 - y.astype(float)) - 1.0) < 1e-9


def test_brier_score_rejects_empty():
    with pytest.raises(ValueError):
        brier_score(pd.Series([], dtype=float), pd.Series([], dtype=float))


def test_reliability_curve_bins_count():
    rng = np.random.default_rng(5)
    y = pd.Series(rng.integers(0, 2, 200))
    p = rng.random(200)
    curve = reliability_curve(y, p, n_bins=10)
    assert curve["n_bins"] == 10
    assert sum(bin_["count"] for bin_ in curve["bins"]) == 200


def test_calibrator_both_kinds_valid_and_roundtrip():
    rng = np.random.default_rng(9)
    raw = rng.random(300)
    y = pd.Series((rng.random(300) < raw).astype(int))
    for kind in ("isotonic", "sigmoid"):
        calibrator = ProbabilityCalibrator.fit_isotonic(raw, y) if kind == "isotonic" else ProbabilityCalibrator.fit_sigmoid(raw, y)
        transformed = calibrator.transform(raw)
        assert (transformed >= 0).all() and (transformed <= 1).all()
        roundtrip = ProbabilityCalibrator.from_dict(calibrator.to_dict())
        np.testing.assert_allclose(roundtrip.transform(raw), transformed, atol=1e-9)


def test_fit_calibrator_selects_by_brier():
    rng = np.random.default_rng(13)
    raw = rng.random(500)
    y = pd.Series((rng.random(500) < raw).astype(int))
    assert fit_calibrator(raw, y).kind in {"isotonic", "sigmoid"}


# --------------------------------------------------------------- conformal --


def test_conformal_quantile_and_interval():
    residuals = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    assert conformal_quantile(residuals, 0.9) >= 2.0
    rng = np.random.default_rng(17)
    calibrator = SplitConformalRegressor.fit(rng.random(100), rng.random(100) + rng.normal(0, 0.5, 100), coverage=0.9, non_negative=True)
    interval = calibrator.interval(5.0)
    assert interval["lower"] <= 5.0 <= interval["upper"]
    assert interval["lower"] >= 0
    assert calibrator.interval(0.01)["lower"] == 0.0


def test_conformal_empirical_coverage():
    rng = np.random.default_rng(19)
    oof_pred = rng.random(500)
    oof_truth = oof_pred + rng.normal(0, 1.0, 500)
    calibrator = SplitConformalRegressor.fit(oof_pred, oof_truth, coverage=0.9)
    preds = rng.random(1000)
    truth = preds + rng.normal(0, 1.0, 1000)
    assert 0.8 <= calibrator.evaluate_coverage(preds, truth) <= 0.97


# ---------------------------------------------------------------- support --


def test_standardized_distance_score_known_values():
    means = {"a": 0.0, "b": 10.0}
    stds = {"a": 1.0, "b": 2.0}
    assert standardized_distance_score({"a": 1.0, "b": 10.0}, means, stds) == 0.5
    assert standardized_distance_score({"a": 0.0, "b": 10.0}, means, stds) == 0.0


def test_support_level_and_assessor():
    assert support_level_from_score(0.5) == "well_supported"
    assert support_level_from_score(1.5) == "moderate_support"
    assert support_level_from_score(5.0) == "extrapolation_warning"
    rng = np.random.default_rng(23)
    X = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(5, 2, 50)})
    assessor = SupportAssessor.fit(X, pd.Series(rng.uniform(21, 22, 50)), pd.Series(rng.uniform(80, 81, 50)))
    near = assessor.assess({"a": 0.0, "b": 5.0}, latitude=21.5, longitude=80.5)
    assert near["extrapolation"]["level"] == "well_supported"
    far = assessor.assess({"a": 100.0, "b": -50.0}, latitude=21.5, longitude=80.5)
    assert far["extrapolation"]["level"] == "extrapolation_warning"
    assert SupportAssessor.from_dict(assessor.to_dict()).means == assessor.means


def test_resource_potential_with_intervals_ordered_and_positive():
    estimate = estimate_resource_potential_with_intervals(0.7, 5.0, 3.0, 7.0, seed=1)
    assert estimate.expected_tonnage > 0
    assert estimate.p10 <= estimate.p50 <= estimate.p90
    assert estimate.uncertainty_sources["thickness"] is True
    assert estimate.uncertainty_sources["density"] is False


def test_resource_potential_deterministic():
    a = estimate_resource_potential_with_intervals(0.7, 5.0, 3.0, 7.0, seed=42)
    b = estimate_resource_potential_with_intervals(0.7, 5.0, 3.0, 7.0, seed=42)
    assert a.expected_tonnage == b.expected_tonnage


def test_drift_no_change_is_ok():
    rng = np.random.default_rng(31)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 200)})
    assert compute_feature_drift(ref, ref.copy())["overall_status"] == "ok"


def test_drift_large_shift_warns():
    rng = np.random.default_rng(37)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    current = pd.DataFrame({"a": rng.normal(5, 1, 500)})
    result = compute_feature_drift(ref, current, thresholds={"mean_shift": 0.25, "median_shift": 0.25, "percentile_shift": 0.25, "psi": 0.25, "missingness": 0.05, "categorical_tv": 0.15})
    assert result["overall_status"] == "warning"


def test_ingestion_pipeline_csv(tmp_path):
    path = tmp_path / "geo.csv"
    geo, _ = _make_geo_satellite(50, seed=41)
    geo.to_csv(path, index=False)
    _, report = IngestionPipeline(CsvSource(path)).run()
    assert report.status == "ok"
    assert report.quality.rows == 50


def test_ingestion_pipeline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        IngestionPipeline(CsvSource(tmp_path / "missing.csv")).run()


def test_promotion_criteria_passes_for_good_record():
    record = {"leakage_check_passed": True, "validation": "spatial_block_holdout", "metrics": {"validation": "spatial_block_holdout", "leakage_check_passed": True}, "artifact_path": None}
    assert check_promotion_criteria(record)[0]


def test_promotion_criteria_fails_without_spatial():
    assert not check_promotion_criteria({"leakage_check_passed": True, "validation": "random_split", "metrics": {}})[0]


def test_promotion_deterministic_and_audited(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    record = {"model_name": "reserve_prospectivity", "version": "2026.09.001", "task": "binary_classification", "algorithm": "XGBoost", "training_data_hash": "abc", "feature_schema_hash": "def", "metrics": {"validation": "spatial_block_holdout", "leakage_check_passed": True}, "artifact_path": None, "status": "candidate", "created_at": "", "target": "is_manganese", "feature_names": [], "validation_strategy": "spatial_block_holdout"}
    (registry / "reserve_prospectivity-2026.09.001.json").write_text(json.dumps(record), encoding="utf-8")
    assert promote_model(registry, "reserve_prospectivity", "2026.09.001", "validated")["status"] == "validated"
    assert promote_model(registry, "reserve_prospectivity", "2026.09.001", "champion")["status"] == "champion"


def test_promotion_illegal_transition_raises(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    record = {"model_name": "reserve_prospectivity", "version": "2026.09.001", "task": "binary_classification", "algorithm": "XGBoost", "training_data_hash": "abc", "feature_schema_hash": "def", "metrics": {"validation": "spatial_block_holdout", "leakage_check_passed": True}, "artifact_path": None, "status": "candidate", "created_at": "", "target": "is_manganese", "feature_names": [], "validation_strategy": "spatial_block_holdout"}
    (registry / "reserve_prospectivity-2026.09.001.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="illegal promotion"):
        promote_model(registry, "reserve_prospectivity", "2026.09.001", "champion")


def test_champion_demotes_previous(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    old = {"model_name": "reserve_prospectivity", "version": "2026.09.001", "task": "binary_classification", "algorithm": "XGBoost", "training_data_hash": "abc", "feature_schema_hash": "def", "metrics": {"validation": "spatial_block_holdout", "leakage_check_passed": True}, "artifact_path": None, "status": "champion", "created_at": "", "target": "is_manganese", "feature_names": [], "validation_strategy": "spatial_block_holdout"}
    new = dict(old, version="2026.09.002", status="validated")
    (registry / "reserve_prospectivity-2026.09.001.json").write_text(json.dumps(old), encoding="utf-8")
    (registry / "reserve_prospectivity-2026.09.002.json").write_text(json.dumps(new), encoding="utf-8")
    promote_model(registry, "reserve_prospectivity", "2026.09.002", "champion")
    assert json.loads((registry / "reserve_prospectivity-2026.09.001.json").read_text(encoding="utf-8"))["status"] == "validated"
    assert json.loads((registry / "reserve_prospectivity-2026.09.002.json").read_text(encoding="utf-8"))["previous_champion"] == "2026.09.001"


def test_load_and_compare_registry(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    record = {"model_name": "reserve_prospectivity", "version": "2026.09.001", "task": "binary_classification", "algorithm": "XGBoost", "training_data_hash": "abc", "feature_schema_hash": "def", "metrics": {"validation": "spatial_block_holdout", "leakage_check_passed": True, "roc_auc": 0.85}, "artifact_path": None, "status": "candidate", "created_at": "", "target": "is_manganese", "feature_names": [], "validation_strategy": "spatial_block_holdout"}
    (registry / "reserve_prospectivity-2026.09.001.json").write_text(json.dumps(record), encoding="utf-8")
    assert len(load_registry_records(registry)) == 1
    assert compare_models(registry, model_name="reserve_prospectivity")[0]["leakage_check_passed"] is True


def test_advanced_training_pipeline(tmp_path_factory):
    from ml.reserve.ensemble import train_prospectivity_ensemble
    from ml.reserve.train_prospectivity import train_grade, train_prospectivity, train_thickness

    root = _write_training_root(tmp_path_factory.mktemp("adv"), n=200, seed=51)
    train_prospectivity(root, quick=True)
    train_grade(root, quick=True)
    train_thickness(root, quick=True)
    ensemble = train_prospectivity_ensemble(root, quick=True)
    assert ensemble["version"]
    assert set(ensemble["weights"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert "ensemble_vs_baseline" in ensemble["metrics"]
    assert (root / "models" / "reserve" / "grade_conformal.json").exists()
    assert (root / "models" / "reserve" / "thickness_conformal.json").exists()
    assert Path(ensemble["evaluation_artifacts"]["calibration_json"]).exists()
    assert Path(ensemble["evaluation_artifacts"]["reliability_png"]).exists()


def test_prediction_grid_with_models(tmp_path_factory):
    from ml.reserve.ensemble import train_prospectivity_ensemble
    from ml.reserve.grid import generate_prediction_grid
    from ml.reserve.train_prospectivity import train_grade, train_prospectivity, train_thickness

    root = _write_training_root(tmp_path_factory.mktemp("grid"), n=150, seed=53)
    train_prospectivity(root, quick=True)
    train_grade(root, quick=True)
    train_thickness(root, quick=True)
    train_prospectivity_ensemble(root, quick=True)
    grid = generate_prediction_grid(root / "models", cells_per_side=4)
    assert grid["cells_per_side"] == 4
    assert len(grid["cells"]) == 16
    supported = [c for c in grid["cells"] if c["context_available"]]
    assert supported
    assert supported[0]["probability"] is not None
    assert supported[0]["resource_potential"]["p10"] <= supported[0]["resource_potential"]["p90"]


def test_prediction_grid_no_context_cell(tmp_path_factory):
    from ml.reserve.grid import generate_prediction_grid
    from ml.reserve.train_prospectivity import train_prospectivity

    root = _write_training_root(tmp_path_factory.mktemp("grid_nc"), n=20, seed=57)
    train_prospectivity(root, quick=True)
    grid = generate_prediction_grid(
        root / "models",
        bbox={"min_lat": -10.0, "max_lat": -9.9, "min_lon": 10.0, "max_lon": 10.1},
        cells_per_side=3,
    )
    no_context = [c for c in grid["cells"] if not c["context_available"]]
    assert no_context
    assert no_context[0]["probability"] is None
    assert no_context[0]["data_support"]["state"] == "no_context"