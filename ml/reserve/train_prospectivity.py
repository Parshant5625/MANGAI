from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from ml.common.registry import ModelVersionRecord, hash_file, hash_schema, write_registry_record
from ml.reserve.evaluate import get_candidates, score_candidates, select_best
from ml.reserve.features import (
    RESERVE_TASKS,
    load_fused_reserve_table,
    prepare_reserve_matrix,
    project_root,
)
from ml.reserve.spatial import (
    assert_no_target_leakage,
    classification_metrics,
    grouped_cv_scores,
    random_holdout_indices,
    regression_metrics,
    spatial_holdout_indices,
)
from ml.reserve.support import SupportAssessor

# Fixed seeds / thread counts keep training reproducible on CPU-only laptops.
RANDOM_STATE = 42
N_JOBS = 4
VERSION_PATTERN = re.compile(r"-(\d{4})\.(\d{2})\.(\d{3})\.json$")


def _artifact_dirs(root: Path) -> tuple[Path, Path, Path]:
    model_dir = root / "models"
    reserve_dir = model_dir / "reserve"
    registry_dir = model_dir / "registry"
    reserve_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, reserve_dir, registry_dir


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def next_version(model_name: str, registry_dir: Path) -> str:
    """Deterministic, monotonically increasing version ``YYYY.MM.NNN``.

    Versions never repeat for a model, so retraining never silently
    overwrites a previous artifact set.
    """
    now = datetime.now(UTC)
    year, month = now.year, now.month
    max_seq = 0
    for path in registry_dir.glob(f"{model_name}-*.json"):
        match = VERSION_PATTERN.search(path.name)
        if not match:
            continue
        rec_year, rec_month, seq = (int(part) for part in match.groups())
        if (rec_year, rec_month) == (year, month):
            max_seq = max(max_seq, seq)
    return f"{year:04d}.{month:02d}.{max_seq + 1:03d}"


def training_data_hash(root: Path) -> str:
    """Stable hash over the exact synthetic training inputs."""
    digest = hashlib.sha256()
    for name in ("geological.csv", "satellite_features.csv"):
        digest.update(hash_file(root / "data/synthetic" / name).encode("utf-8"))
    return digest.hexdigest()[:16]


def train_prospectivity(root: Path | None = None, quick: bool = False) -> dict:
    return _train_task(root, task_name="prospectivity", quick=quick)


def train_grade(root: Path | None = None, quick: bool = False) -> dict:
    return _train_task(root, task_name="grade", quick=quick)


def train_thickness(root: Path | None = None, quick: bool = False) -> dict:
    return _train_task(root, task_name="thickness", quick=quick)


def _train_task(root: Path | None, task_name: str, quick: bool = False) -> dict:
    """Train one reserve baseline end-to-end.

    Steps: load fused data -> build task features -> leakage assertions ->
    spatial holdout + grouped CV candidate comparison -> select winner ->
    random-split diagnostic (recorded, never primary) -> persist versioned
    artifacts + metadata -> write registry record.
    """
    root = root or project_root()
    spec = RESERVE_TASKS[task_name]
    df = load_fused_reserve_table(root)

    X = prepare_reserve_matrix(
        df,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    y = df[spec.target]
    y = y.astype(int) if spec.kind == "classification" else y.astype(float)

    # MANDATORY leakage gates: training fails loudly if any target-derived
    # column entered the feature matrix.
    assert_no_target_leakage(X, spec.forbidden_columns, context=task_name)
    from ml.common.validation import check_leakage

    leakage_report = check_leakage(X, [spec.target], context=f"reserve/{task_name}")
    leakage_report.raise_for_errors()

    train_idx, test_idx, groups = spatial_holdout_indices(df)
    comparison = score_candidates(
        spec.kind, X, y, groups, train_idx, test_idx, quick=quick
    )
    best_name = select_best(comparison, spec.kind)
    model = get_candidates(spec.kind, quick=quick)[best_name]()

    # Spatial-block holdout is the PRIMARY validation.
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    if spec.kind == "classification":
        probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]
        metrics = classification_metrics(y.iloc[test_idx], probabilities)
    else:
        metrics = regression_metrics(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
    cv_scores = grouped_cv_scores(
        get_candidates(spec.kind, quick=quick)[best_name], X, y, groups, spec.kind
    )
    metrics.update(cv_scores)

    # Random i.i.d. split — DIAGNOSTIC ONLY (quantifies spatial optimism).
    random_train_idx, random_test_idx = random_holdout_indices(df)
    diagnostic_model = get_candidates(spec.kind, quick=quick)[best_name]()
    diagnostic_model.fit(X.iloc[random_train_idx], y.iloc[random_train_idx])
    if spec.kind == "classification":
        random_metrics = classification_metrics(
            y.iloc[random_test_idx], diagnostic_model.predict_proba(X.iloc[random_test_idx])[:, 1]
        )
    else:
        random_metrics = regression_metrics(
            y.iloc[random_test_idx], diagnostic_model.predict(X.iloc[random_test_idx])
        )
    random_split_diagnostic = {
        "metrics": random_metrics,
        "note": (
            "Random i.i.d. split allows spatially adjacent samples to leak between "
            "train and validation. Reported for comparison only — the spatial-block "
            "holdout above is the primary validation."
        ),
    }

    feature_columns = list(X.columns)
    metrics["validation"] = "spatial_block_holdout"
    metrics["validation_details"] = {
        "strategy": "GroupShuffleSplit over 5x5 latitude/longitude spatial blocks",
        "primary": True,
        "held_out_blocks": int(groups.iloc[test_idx].nunique()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "random_state": RANDOM_STATE,
    }
    metrics["candidate_comparison"] = comparison
    metrics["selected_algorithm"] = best_name
    metrics["selection_metric"] = spec.primary_metric
    metrics["random_split_diagnostic"] = random_split_diagnostic
    metrics["target"] = spec.target
    metrics["leakage_exclusions"] = spec.forbidden_columns
    metrics["leakage_check_passed"] = bool(leakage_report.valid)
    metrics["feature_names"] = feature_columns
    metrics["training_rows"] = int(len(df))
    metrics["synthetic_data"] = True

    # Conformal prediction intervals for regression tasks (grouped OOF, no leakage).
    conformal_state = None
    if spec.kind == "regression":
        conformal_state = _fit_conformal(
            get_candidates(spec.kind, quick=quick)[best_name],
            X,
            y,
            groups,
            train_idx,
            non_negative=(spec.target == "ore_thickness_m"),
        )
        metrics["conformal"] = {
            "method": "split_conformal",
            "coverage": conformal_state.coverage,
            "n_calibration": conformal_state.n_calibration,
            "non_negative": conformal_state.non_negative,
            "empirical_coverage": conformal_state.evaluate_coverage(
                model.predict(X.iloc[test_idx]), y.iloc[test_idx]
            ),
            "note": "Coverage is empirical on the synthetic spatial holdout under grouped-exchangeability.",
        }

    # Data-support statistics for inference-time extrapolation detection.
    support_assessor = SupportAssessor.fit(X, df["latitude"], df["longitude"])
    metrics["support_stats"] = support_assessor.to_dict()

    result = _persist_model(
        root,
        model_name=f"reserve_{task_name}",
        task="binary_classification" if spec.kind == "classification" else "regression",
        spec=spec,
        algorithm_label=_algorithm_label(best_name),
        model=model,
        feature_columns=feature_columns,
        metrics=metrics,
        quick=quick,
    )

    # Persist conformal sidecar (regression only) alongside the model artifacts.
    if conformal_state is not None:
        model_dir = root / "models"
        reserve_dir = model_dir / "reserve"
        sidecar = conformal_state.to_dict()
        for directory in (reserve_dir / "versions" / result["version"], reserve_dir):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{spec.name}_conformal.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return result


def _fit_conformal(factory, X, y, groups, train_idx, *, non_negative: bool):
    """Grouped-OOF conformal calibration on the training blocks (no leakage)."""
    from sklearn.model_selection import GroupKFold

    from ml.reserve.conformal import SplitConformalRegressor

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    n_splits = min(5, max(2, int(groups_train.nunique())))
    oof = np.zeros(len(y_train))
    fold = GroupKFold(n_splits=n_splits)
    for fit_idx, val_idx in fold.split(X_train, y_train, groups_train):
        model = factory()
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof[val_idx] = model.predict(X_train.iloc[val_idx])
    return SplitConformalRegressor.fit(oof, y_train, coverage=0.9, non_negative=non_negative)


def _algorithm_label(candidate_name: str) -> str:
    return {
        "logistic_regression": "LogisticRegression",
        "ridge": "Ridge",
        "random_forest": "RandomForest",
        "xgboost": "XGBoost",
    }.get(candidate_name, candidate_name)


def _persist_model(
    root: Path,
    model_name: str,
    task: str,
    spec,
    algorithm_label: str,
    model,
    feature_columns: list[str],
    metrics: dict,
    quick: bool = False,
) -> dict:
    model_dir, reserve_dir, registry_dir = _artifact_dirs(root)
    version = next_version(model_name, registry_dir)
    stem = f"{spec.name}_{algorithm_label.lower()}"
    versioned_dir = reserve_dir / "versions" / version
    versioned_dir.mkdir(parents=True, exist_ok=True)

    artifact = versioned_dir / f"{stem}.json"
    joblib.dump(model, versioned_dir / f"{stem}_model.joblib")
    joblib.dump(feature_columns, versioned_dir / f"{stem}_features.pkl")
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        model.save_model(artifact)  # native XGBoost format for the serving path
        registry_artifact = artifact
    else:
        # Non-XGBoost winners are served from the joblib bundle.
        registry_artifact = versioned_dir / f"{stem}_model.joblib"

    data_hash = training_data_hash(root)
    meta = {
        "model_name": model_name,
        "version": version,
        "task": task,
        "task_kind": spec.kind,
        "algorithm": algorithm_label,
        "target": spec.target,
        "prediction_type": "manganese_occurrence_probability"
        if spec.kind == "classification"
        else spec.target,
        "feature_names": feature_columns,
        "feature_schema_hash": hash_schema({"features": feature_columns})[:16],
        "training_data_hash": data_hash,
        "training_data": ["data/synthetic/geological.csv", "data/synthetic/satellite_features.csv"],
        "validation_strategy": metrics["validation"],
        "metrics": metrics,
        "random_seed": RANDOM_STATE,
        "quick_mode": quick,
        "synthetic_data": True,
        "boundary_notice": "PROTOTYPE resource intelligence on synthetic demo data. NOT official mineral reserves; NOT field-validated.",
        "status": "candidate",
    }
    _save_json(versioned_dir / f"{stem}_meta.json", meta)

    # Latest fixed-path serving copies (consumed by backend inference).
    latest_artifact = reserve_dir / f"{spec.name}_xgboost.json"
    if isinstance(model, (XGBClassifier, XGBRegressor)):
        model.save_model(latest_artifact)
    joblib.dump(model, reserve_dir / f"{spec.name}_model.joblib")
    joblib.dump(feature_columns, reserve_dir / f"{spec.name}_xgboost_features.pkl")
    joblib.dump(feature_columns, reserve_dir / f"{spec.name}_features.pkl")
    _save_json(reserve_dir / f"{spec.name}_metrics.json", metrics)
    _save_json(reserve_dir / f"{spec.name}_meta.json", meta)

    if spec.name == "prospectivity":
        # Legacy root-level artifacts kept for older consumers.
        if isinstance(model, (XGBClassifier, XGBRegressor)):
            model.save_model(model_dir / "reserve_xgboost.json")
        joblib.dump(model, model_dir / "reserve_model.joblib")
        joblib.dump(feature_columns, model_dir / "reserve_features.pkl")
        _save_json(model_dir / "reserve_metrics.json", metrics)
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = [0.0] * len(feature_columns)
        importance = pd.DataFrame(
            {"feature": feature_columns, "importance": importances}
        ).sort_values("importance", ascending=False)
        importance.to_csv(model_dir / "reserve_feature_importance.csv", index=False)

    record = ModelVersionRecord(
        model_name=model_name,
        version=version,
        task=task,
        algorithm=algorithm_label,
        training_data_hash=data_hash,
        feature_schema_hash=hash_schema({"features": feature_columns})[:16],
        metrics=metrics,
        artifact_path=str(registry_artifact.relative_to(root)).replace("\\", "/"),
        status="candidate",
        target=spec.target,
        feature_names=feature_columns,
        validation_strategy=metrics["validation"],
    )
    registry_path = write_registry_record(record, registry_dir)
    return {
        "model_name": model_name,
        "version": version,
        "algorithm": algorithm_label,
        "selected_from": list(metrics["candidate_comparison"].keys()),
        "metrics": metrics,
        "artifact_path": record.artifact_path,
        "registry_path": str(registry_path.relative_to(root)).replace("\\", "/"),
    }


if __name__ == "__main__":
    print(train_prospectivity())
