"""Phase 3 tests: reserve AI baseline.

Covers: feature construction, target-leakage protection, spatial validation,
preprocessing, training of all three tasks, serialization/loading, inference,
missing-model behaviour, deterministic retraining, and model comparison.
Uses small deterministic synthetic fixtures — no expensive training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.reserve.features import (
    RESERVE_CATEGORICAL_FEATURES,
    RESERVE_FORBIDDEN_COLUMNS,
    RESERVE_NUMERICAL_FEATURES,
    RESERVE_TASKS,
    prepare_reserve_matrix,
)
from ml.reserve.spatial import (
    assert_no_target_leakage,
    classification_metrics,
    random_holdout_indices,
    regression_metrics,
    spatial_block_id,
    spatial_holdout_indices,
)
from ml.reserve.train_prospectivity import next_version, train_grade, train_prospectivity, train_thickness

FORMATION_VALUES = ["Manganiferous_Formation", "Gondite", "Phyllite", "Banded_Iron_Formation"]


def _make_geo_satellite(n: int = 600, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic synthetic reserve tables with learnable spatial structure."""
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(21.0, 22.0, n)
    longitude = rng.uniform(80.0, 81.0, n)
    # Manganese favourability increases to the north-east (learnable, spatial).
    logit = 4.0 * (latitude - 21.0) + 4.0 * (longitude - 80.0) - 3.0
    probability = 1.0 / (1.0 + np.exp(-logit))
    is_manganese = rng.binomial(1, probability)

    formation = np.array(FORMATION_VALUES, dtype=object)[rng.integers(0, len(FORMATION_VALUES), n)]
    # Manganiferous formation is more common where manganese occurs.
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


def _write_training_root(root: Path, n: int = 600, seed: int = 7) -> Path:
    geo, sat = _make_geo_satellite(n, seed)
    synthetic = root / "data" / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    geo.to_csv(synthetic / "geological.csv", index=False)
    sat.to_csv(synthetic / "satellite_features.csv", index=False)
    return root


@pytest.fixture(scope="module")
def trained_root(tmp_path_factory) -> Path:
    """Train all three baselines once per module on a small fixture."""
    root = _write_training_root(tmp_path_factory.mktemp("reserve_train"))
    train_prospectivity(root, quick=True)
    train_grade(root, quick=True)
    train_thickness(root, quick=True)
    return root


# ---------------------------------------------------------------- features --


def test_task_specs_cover_three_tasks():
    assert set(RESERVE_TASKS) == {"prospectivity", "grade", "thickness"}
    assert RESERVE_TASKS["prospectivity"].kind == "classification"
    assert RESERVE_TASKS["prospectivity"].target == "is_manganese"
    assert RESERVE_TASKS["grade"].target == "mn_pct"
    assert RESERVE_TASKS["thickness"].target == "ore_thickness_m"


def test_task_features_are_explicit_not_all_numeric():
    """Feature lists are curated — never 'every numeric column'."""
    for spec in RESERVE_TASKS.values():
        assert set(spec.numerical_features) <= set(RESERVE_NUMERICAL_FEATURES)
        assert set(spec.categorical_features) <= set(RESERVE_CATEGORICAL_FEATURES)


def test_reserve_matrix_excludes_all_forbidden_columns(trained_root):
    df = pd.read_csv(trained_root / "data" / "synthetic" / "geological.csv")
    sat = pd.read_csv(trained_root / "data" / "synthetic" / "satellite_features.csv")
    fused = df.merge(sat, on=["sample_id", "latitude", "longitude"])
    matrix = prepare_reserve_matrix(fused)
    for column in RESERVE_FORBIDDEN_COLUMNS:
        assert column not in matrix.columns


# ----------------------------------------------------------------- leakage --


def test_assert_no_target_leakage_raises():
    X = pd.DataFrame({"ndvi": [0.1, 0.2], "mn_pct": [10.0, 12.0]})
    with pytest.raises(ValueError, match="mn_pct"):
        assert_no_target_leakage(X, RESERVE_FORBIDDEN_COLUMNS, context="grade")


def test_assert_no_target_leakage_passes_clean_matrix():
    X = pd.DataFrame({"ndvi": [0.1, 0.2], "slope_deg": [5.0, 8.0]})
    assert_no_target_leakage(X, RESERVE_FORBIDDEN_COLUMNS, context="grade")


# ------------------------------------------------------------ spatial split --


def test_spatial_split_is_deterministic_and_disjoint():
    geo, sat = _make_geo_satellite(200, seed=3)
    df = geo.merge(sat, on=["sample_id", "latitude", "longitude"])
    train_a, test_a, groups_a = spatial_holdout_indices(df)
    train_b, test_b, groups_b = spatial_holdout_indices(df)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
    pd.testing.assert_series_equal(groups_a, groups_b)
    assert set(train_a) & set(test_a) == set()


def test_spatial_split_has_no_group_overlap():
    geo, sat = _make_geo_satellite(400, seed=5)
    df = geo.merge(sat, on=["sample_id", "latitude", "longitude"])
    train_idx, test_idx, groups = spatial_holdout_indices(df)
    train_blocks = set(groups.iloc[train_idx])
    test_blocks = set(groups.iloc[test_idx])
    assert train_blocks & test_blocks == set()
    assert test_blocks, "spatial holdout must hold out at least one block"


def test_random_split_diagnostic_leaks_across_blocks():
    """The random split (diagnostic) shares spatial blocks across the split —
    exactly the leakage the primary spatial split prevents."""
    geo, sat = _make_geo_satellite(400, seed=5)
    df = geo.merge(sat, on=["sample_id", "latitude", "longitude"])
    groups = spatial_block_id(df, blocks=5)
    train_idx, test_idx = random_holdout_indices(df)
    assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])


# --------------------------------------------------------- classification ----


def test_classification_metrics_include_confusion_matrix():
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    probabilities = pd.Series([0.1, 0.2, 0.8, 0.9, 0.4, 0.3])
    metrics = classification_metrics(y_true, probabilities)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert key in metrics
    matrix = metrics["confusion_matrix"]
    assert matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"] == len(y_true)


# ------------------------------------------------------------- training -----


def test_prospectivity_training_artifacts_and_metrics(trained_root):
    result = train_prospectivity(trained_root, quick=True)
    metrics = result["metrics"]
    assert result["algorithm"] in {"LogisticRegression", "RandomForest", "XGBoost"}
    assert metrics["validation"] == "spatial_block_holdout"
    matrix = metrics["confusion_matrix"]
    assert matrix["tn"] + matrix["fp"] + matrix["fn"] + matrix["tp"] > 0
    assert metrics["target"] == "is_manganese"
    assert metrics["leakage_check_passed"] is True
    assert set(metrics["candidate_comparison"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert metrics["random_split_diagnostic"]["note"]
    registry_file = (
        trained_root / "models" / "registry" / f"reserve_prospectivity-{result['version']}.json"
    )
    assert registry_file.exists()
    record = json.loads(registry_file.read_text(encoding="utf-8"))
    assert record["feature_names"] == metrics["feature_names"]
    assert record["training_data_hash"]


def test_grade_training_metrics(trained_root):
    result = train_grade(trained_root, quick=True)
    metrics = result["metrics"]
    for key in ("mae", "rmse", "r2", "cv_mae", "cv_rmse", "cv_r2"):
        assert key in metrics
    assert metrics["target"] == "mn_pct"
    assert metrics["leakage_check_passed"] is True


def test_thickness_training_metrics(trained_root):
    result = train_thickness(trained_root, quick=True)
    metrics = result["metrics"]
    for key in ("mae", "rmse", "r2"):
        assert key in metrics
    assert metrics["target"] == "ore_thickness_m"
    assert metrics["leakage_exclusions"] == RESERVE_FORBIDDEN_COLUMNS


# --------------------------------------------- serialization / inference ----


def test_model_serialization_roundtrip_and_metadata(trained_root):
    from ml.reserve.inference import load_model_bundle, load_model_metadata

    artifact = trained_root / "models" / "reserve" / "prospectivity_model.joblib"
    assert artifact.exists()
    model, columns = load_model_bundle(artifact, task="classification")
    assert isinstance(columns, list) and columns
    assert hasattr(model, "predict_proba")
    meta = load_model_metadata(artifact)
    assert meta is not None
    assert meta["model_name"] == "reserve_prospectivity"
    assert meta["synthetic_data"] is True
    assert meta["boundary_notice"]


def test_inference_predicts_from_raw_frame(trained_root):
    from ml.reserve.inference import maybe_predict_regressor, predict_with_model

    geo, sat = _make_geo_satellite(12, seed=11)
    frame = geo.merge(sat, on=["sample_id", "latitude", "longitude"]).drop(
        columns=["is_manganese", "mn_pct", "fe_pct", "sio2_pct", "ore_thickness_m"]
    )
    model_path = trained_root / "models" / "reserve" / "prospectivity_model.joblib"
    probabilities = predict_with_model(frame, model_path, "classification")
    assert probabilities.between(0, 1).all()
    assert len(probabilities) == len(frame)

    grade_path = trained_root / "models" / "reserve" / "grade_model.joblib"
    grade = maybe_predict_regressor(frame, grade_path)
    assert grade is not None
    assert np.isfinite(grade).all()
    assert (grade >= 0).all()


def test_missing_model_behaviour(tmp_path):
    from ml.reserve.inference import maybe_predict_regressor, predict_prospectivity_frame

    frame = pd.DataFrame({"ndvi": [0.3]})
    assert maybe_predict_regressor(frame, tmp_path / "missing.json") is None
    with pytest.raises(FileNotFoundError):
        predict_prospectivity_frame(frame, tmp_path)


# ------------------------------------------------------------ determinism ----


def test_training_is_reproducible(tmp_path_factory):
    """Same data + same seed => identical metrics, same selected algorithm."""
    root_a = _write_training_root(tmp_path_factory.mktemp("repro_a"), n=300, seed=21)
    root_b = _write_training_root(tmp_path_factory.mktemp("repro_b"), n=300, seed=21)
    result_a = train_prospectivity(root_a, quick=True)
    result_b = train_prospectivity(root_b, quick=True)
    for key in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
        assert round(result_a["metrics"][key], 10) == round(result_b["metrics"][key], 10)
    assert result_a["metrics"]["selected_algorithm"] == result_b["metrics"]["selected_algorithm"]


def test_versions_never_collide(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    version_one = next_version("reserve_prospectivity", registry)
    assert version_one.endswith(".001")
    (registry / f"reserve_prospectivity-{version_one}.json").write_text("{}", encoding="utf-8")
    version_two = next_version("reserve_prospectivity", registry)
    assert version_two.endswith(".002")
    assert version_two != version_one


# ------------------------------------------------- regression metric sanity --


def test_regression_metrics_bounds():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert abs(metrics["r2"] - 1.0) < 1e-12
