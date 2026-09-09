"""Phase 4.1 validation-integrity audit tests.

Proves that the reserve ML pipelines keep a genuinely untouched final spatial
test set:

1. final test groups do not overlap development groups
2. calibration groups do not overlap final test groups
3. ensemble weights are not fitted using final test
4. calibrator is not fitted using final test
5. conformal quantile is not fitted using final test
6. preprocessing does not fit on final test
7. target leakage remains blocked
8. final test metrics are calculated only from final test
9. probability remains [0,1]
10. prediction intervals satisfy lower <= prediction <= upper
11. P10 <= P50 <= P90

All fixtures are small and synthetic; training runs in ``quick`` mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.reserve.calibration import fit_calibrator
from ml.reserve.conformal import SplitConformalRegressor
from ml.reserve.features import (
    RESERVE_FORBIDDEN_COLUMNS,
    RESERVE_TASKS,
    load_fused_reserve_table,
    prepare_reserve_matrix,
)
from ml.reserve.resource_estimator import estimate_resource_potential_with_intervals
from ml.reserve.spatial import spatial_dev_test_split

FORMATION_VALUES = ["Manganiferous_Formation", "Gondite", "Phyllite", "Banded_Iron_Formation"]


def _make_geo_satellite(n: int = 300, seed: int = 61) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _write_training_root(root: Path, n: int = 300, seed: int = 61) -> Path:
    geo, sat = _make_geo_satellite(n, seed)
    synthetic = root / "data" / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    geo.to_csv(synthetic / "geological.csv", index=False)
    sat.to_csv(synthetic / "satellite_features.csv", index=False)
    return root


def _fused_and_matrix(root: Path, task_name: str = "prospectivity"):
    df = load_fused_reserve_table(root)
    spec = RESERVE_TASKS[task_name]
    X = prepare_reserve_matrix(
        df,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    y = df[spec.target]
    y = y.astype(int) if spec.kind == "classification" else y.astype(float)
    return df, spec, X, y


@pytest.fixture(scope="module")
def audited_root(tmp_path_factory) -> Path:
    """Train all Phase 4 artifacts once per module on a small fixture."""
    from ml.reserve.ensemble import train_prospectivity_ensemble
    from ml.reserve.train_prospectivity import train_grade, train_prospectivity, train_thickness

    root = _write_training_root(tmp_path_factory.mktemp("audit"), n=300, seed=61)
    train_prospectivity(root, quick=True)
    train_grade(root, quick=True)
    train_thickness(root, quick=True)
    train_prospectivity_ensemble(root, quick=True)
    return root


def _load_metrics(root: Path, name: str) -> dict:
    return json.loads((root / "models" / "reserve" / f"{name}_metrics.json").read_text(encoding="utf-8"))


# 1. final test groups do not overlap development groups
def test_final_test_groups_do_not_overlap_development_groups(audited_root):
    metrics = _load_metrics(audited_root, "prospectivity")
    assert metrics["group_overlap"] == []
    assert set(metrics["development_spatial_groups"]) & set(metrics["final_test_spatial_groups"]) == set()
    assert metrics["leakage_check_passed"] is True
    assert metrics["final_evaluation_isolation"] is True
    assert metrics["development_sample_count"] + metrics["final_test_sample_count"] == metrics["training_rows"]


# 2. calibration groups do not overlap final test groups
def test_calibration_groups_do_not_overlap_final_test(audited_root):
    ensemble_meta = json.loads(
        (audited_root / "models" / "reserve" / "prospectivity_ensemble_meta.json").read_text(encoding="utf-8")
    )
    assert set(ensemble_meta["development_spatial_groups"]) & set(ensemble_meta["final_test_spatial_groups"]) == set()
    assert "development" in ensemble_meta["calibration_method"]
    # Conformal calibration on the regression tasks likewise uses only dev rows.
    grade_metrics = _load_metrics(audited_root, "grade")
    thickness_metrics = _load_metrics(audited_root, "thickness")
    for metrics in (grade_metrics, thickness_metrics):
        conformal = metrics["conformal"]
        assert conformal["calibration_source"] == "development_set_oof_residuals_only"
        assert conformal["calibration_sample_count"] <= metrics["development_sample_count"]
        assert conformal["calibration_sample_count"] > 0
        assert metrics["group_overlap"] == []


# 3. ensemble weights are not fitted using final test
def test_ensemble_weights_fitted_only_on_development(audited_root):
    from ml.reserve.ensemble import _grouped_oof_probabilities, compute_member_weights, train_prospectivity_ensemble
    from ml.reserve.evaluate import get_candidates

    df, spec, X, y = _fused_and_matrix(audited_root, "prospectivity")
    dev_idx, test_idx, groups = spatial_dev_test_split(df)

    def _dev_weights(target: pd.Series) -> dict[str, float]:
        oof = {}
        for name, factory in get_candidates("classification", quick=True).items():
            oof[name] = _grouped_oof_probabilities(factory, X, target, groups, dev_idx)
        return compute_member_weights(oof, target.iloc[dev_idx].reset_index(drop=True))

    reference = _dev_weights(y)
    assert sum(reference.values()) > 0

    # Perturbing ONLY the final test labels cannot change the fitted weights,
    # because OOF fitting only ever consumes DEVELOPMENT rows.
    y_perturbed = y.copy()
    y_perturbed.iloc[test_idx] = 1 - y_perturbed.iloc[test_idx]
    assert _dev_weights(y_perturbed) == reference

    # Persisted weights match a from-scratch development-only computation.
    result = train_prospectivity_ensemble(audited_root, quick=True)
    for name in reference:
        assert abs(reference[name] - result["weights"][name]) < 1e-12, f"{name} weight leaked test info"
    meta = json.loads(
        (audited_root / "models" / "reserve" / "prospectivity_ensemble_meta.json").read_text(encoding="utf-8")
    )
    assert "development set only" in meta["ensemble_weight_selection_method"]


# 4. calibrator is not fitted using final test
def test_calibrator_fitted_only_on_development(audited_root):
    from ml.reserve.ensemble import ReserveEnsemble, _grouped_oof_probabilities, compute_member_weights
    from ml.reserve.evaluate import get_candidates

    df, spec, X, y = _fused_and_matrix(audited_root, "prospectivity")
    dev_idx, test_idx, groups = spatial_dev_test_split(df)
    y_dev = y.iloc[dev_idx].reset_index(drop=True)

    def _dev_raw(target: pd.Series) -> np.ndarray:
        oof = {
            name: _grouped_oof_probabilities(factory, X, target, groups, dev_idx)
            for name, factory in get_candidates("classification", quick=True).items()
        }
        weights = compute_member_weights(oof, target.iloc[dev_idx].reset_index(drop=True))
        return sum(weights[name] * oof[name] for name in weights)

    reference_raw = _dev_raw(y)
    reference = fit_calibrator(reference_raw, y_dev)

    # Perturbing ONLY the final test labels cannot change the fitted calibrator.
    y_perturbed = y.copy()
    y_perturbed.iloc[test_idx] = 1 - y_perturbed.iloc[test_idx]
    perturbed_raw = _dev_raw(y_perturbed)
    perturbed = fit_calibrator(perturbed_raw, y_dev)
    np.testing.assert_allclose(reference.transform(reference_raw), perturbed.transform(perturbed_raw), atol=1e-9)

    # The persisted calibrator must match the development-only fit.
    ensemble = ReserveEnsemble.load(audited_root / "models" / "reserve" / "prospectivity_ensemble_model.joblib")
    np.testing.assert_allclose(
        ensemble.calibrator.transform(reference_raw),
        reference.transform(reference_raw),
        atol=1e-9,
    )


# 5. conformal quantile is not fitted using final test
def test_conformal_quantile_fitted_only_on_development(audited_root):
    from ml.reserve.evaluate import get_candidates
    from ml.reserve.train_prospectivity import _fit_conformal

    df, spec, X, y = _fused_and_matrix(audited_root, "grade")
    dev_idx, test_idx, groups = spatial_dev_test_split(df)
    metrics = _load_metrics(audited_root, "grade")
    factory = get_candidates("regression", quick=True)[metrics["selected_algorithm"]]

    reference = _fit_conformal(factory, X, y, groups, dev_idx, non_negative=False)
    assert reference.n_calibration == len(dev_idx)

    # Perturbing ONLY the final test labels cannot change the quantile.
    y_perturbed = y.copy()
    y_perturbed.iloc[test_idx] = -1.0  # wildly different test target
    perturbed = _fit_conformal(factory, X, y_perturbed, groups, dev_idx, non_negative=False)
    assert abs(perturbed.quantile - reference.quantile) < 1e-12

    # The persisted sidecar quantile equals the development-only computation.
    sidecar = json.loads((audited_root / "models" / "reserve" / "grade_conformal.json").read_text(encoding="utf-8"))
    assert abs(sidecar["quantile"] - reference.quantile) < 1e-12
    assert sidecar["n_calibration"] == len(dev_idx)


# 6. preprocessing does not fit on final test
def test_preprocessing_and_support_stats_fit_only_on_development(audited_root):
    metrics = _load_metrics(audited_root, "prospectivity")
    df, spec, X, y = _fused_and_matrix(audited_root, "prospectivity")
    dev_idx, test_idx, groups = spatial_dev_test_split(df)

    # One-hot alignment is schema-based (stateless): a test-only matrix cannot
    # introduce columns absent from the training schema.
    test_frame = df.iloc[test_idx].reset_index(drop=True)
    test_matrix = prepare_reserve_matrix(
        test_frame,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
        feature_columns=metrics["feature_names"],
    )
    assert set(test_matrix.columns) <= set(metrics["feature_names"])

    # The persisted data-support reference stats must equal DEVELOPMENT-only
    # statistics (not full-data or test-inclusive statistics).
    support = metrics["support_stats"]
    dev_matrix = prepare_reserve_matrix(
        df.iloc[dev_idx].reset_index(drop=True),
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    for column in dev_matrix.columns:
        assert abs(support["means"][column] - float(dev_matrix[column].mean())) < 1e-9, (
            f"support mean for {column} leaked test information"
        )


# 7. target leakage remains blocked
def test_target_leakage_remains_blocked(audited_root):
    df, spec, X, y = _fused_and_matrix(audited_root, "prospectivity")
    assert not (set(RESERVE_FORBIDDEN_COLUMNS) & set(X.columns))
    metrics = _load_metrics(audited_root, "prospectivity")
    assert metrics["leakage_check_passed"] is True
    assert metrics["leakage_exclusions"] == RESERVE_FORBIDDEN_COLUMNS


# 8. final test metrics are calculated only from final test
def test_final_metrics_computed_only_on_final_test(audited_root):
    metrics = _load_metrics(audited_root, "prospectivity")
    n_test = metrics["final_test_sample_count"]
    matrix = metrics["confusion_matrix"]
    assert matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"] == n_test, (
        "confusion matrix must cover exactly the final test rows"
    )
    # Selection happens on development CV scores, never on the test set.
    assert metrics["selection_metric"].startswith("cv_")
    assert metrics["selection_protocol"] == "grouped_cv_on_development_set_only"
    # Development and final metrics are stored as separate records.
    grade_metrics = _load_metrics(audited_root, "grade")
    assert set(grade_metrics["development_metrics"]) == {"cv_mae", "cv_rmse", "cv_r2"}
    ensemble_meta = json.loads(
        (audited_root / "models" / "reserve" / "prospectivity_ensemble_meta.json").read_text(encoding="utf-8")
    )
    assert ensemble_meta["metrics"]["final_test_sample_count"] == n_test
    assert "ensemble_oof_roc_auc" in ensemble_meta["metrics"]["development_metrics"]


# 9. probability remains [0, 1]
def test_probabilities_bounded_unit_interval(audited_root):
    from ml.reserve.ensemble import ReserveEnsemble
    from ml.reserve.inference import predict_with_model

    df = load_fused_reserve_table(audited_root)
    probs = predict_with_model(
        df, audited_root / "models" / "reserve" / "prospectivity_model.joblib", "classification"
    )
    assert probs.between(0, 1).all()

    X = prepare_reserve_matrix(df)
    ensemble = ReserveEnsemble.load(audited_root / "models" / "reserve" / "prospectivity_ensemble_model.joblib")
    calibrated = ensemble.predict_proba(X)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()
    raw = ensemble.raw_probabilities(X)
    assert (raw >= 0).all() and (raw <= 1).all()


# 10. prediction intervals satisfy lower <= prediction <= upper
def test_prediction_intervals_contain_point_prediction(audited_root):
    for name in ("grade", "thickness"):
        sidecar = json.loads(
            (audited_root / "models" / "reserve" / f"{name}_conformal.json").read_text(encoding="utf-8")
        )
        regressor = SplitConformalRegressor.from_dict(sidecar)
        for prediction in (0.1, 1.0, 5.0, 25.0):
            interval = regressor.interval(prediction)
            assert interval["lower"] <= prediction <= interval["upper"], name
            if regressor.non_negative:
                assert interval["lower"] >= 0, name


# 11. P10 <= P50 <= P90 and resource >= 0
def test_resource_uncertainty_invariants(audited_root):
    estimate = estimate_resource_potential_with_intervals(0.6, 5.0, 3.0, 7.0)
    assert estimate.p10 <= estimate.p50 <= estimate.p90
    assert estimate.expected_tonnage >= 0
    assert estimate.p10 >= 0
    assert estimate.uncertainty_sources["thickness"] is True
    # No fabricated uncertainty: when no probability/density std is passed,
    # those sources are reported as fixed (False).
    assert estimate.uncertainty_sources["probability"] is False
    assert estimate.uncertainty_sources["density"] is False