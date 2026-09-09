"""Weighted soft-voting ensemble for manganese prospectivity (Phase 4).

Strategy (defensible and simple — complexity only where it measurably helps):

1. Base learners: the SAME three Phase 3 candidates (Logistic Regression,
   Random Forest, XGBoost), trained on the TRAINING spatial blocks.
2. Weights: proportional to ``max(roc_auc_oof - 0.5, epsilon)`` where each
   member's ROC-AUC is computed on OUT-OF-FOLD spatial predictions
   (``GroupKFold`` over the training blocks). A member barely better than
   random gets almost no weight; nothing is hard-coded.
3. Calibration: a monotone calibrator (isotonic or sigmoid, chosen by Brier
   score) is fitted on the pooled out-of-fold ensemble probabilities —
   see ``ml.reserve.calibration`` for the leakage-avoidance rationale.
4. Selection: the ensemble is compared against the best single baseline under
   the SAME spatial holdout. The ensemble is only preferred when its primary
   metric (ROC-AUC) is at least as good; complexity must justify itself.
   Both ROC-AUC/PR-AUC/F1/precision/recall and Brier scores are reported.

All outputs are prototype intelligence on SYNTHETIC data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ml.reserve.calibration import (
    ProbabilityCalibrator,
    brier_score,
    fit_calibrator,
    reliability_curve,
    save_evaluation_artifacts,
)
from ml.reserve.evaluate import get_candidates
from ml.reserve.features import (
    RESERVE_TASKS,
    load_fused_reserve_table,
    prepare_reserve_matrix,
    project_root,
)
from ml.reserve.spatial import (
    assert_no_target_leakage,
    classification_metrics,
    group_split_report,
    spatial_dev_test_split,
)
from ml.reserve.support import SupportAssessor


def _grouped_oof_probabilities(
    factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series, train_idx
) -> np.ndarray:
    """Out-of-fold probabilities on the training blocks (GroupKFold)."""
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    n_splits = min(5, max(2, int(groups_train.nunique())))
    oof = np.zeros(len(y_train))
    fold = GroupKFold(n_splits=n_splits)
    for fit_idx, val_idx in fold.split(X_train, y_train, groups_train):
        model = factory()
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    return oof


def compute_member_weights(
    oof_probabilities: dict[str, np.ndarray], y_oof: pd.Series
) -> dict[str, float]:
    """Weights proportional to max(OOF ROC-AUC - 0.5, epsilon), normalized."""
    epsilon = 0.01
    raw: dict[str, float] = {}
    for name, probabilities in oof_probabilities.items():
        roc = float(roc_auc_score(y_oof, probabilities))
        raw[name] = max(roc - 0.5, epsilon)
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


class ReserveEnsemble:
    """Weighted soft-voting ensemble with a monotone probability calibrator."""

    def __init__(
        self,
        members: dict[str, Any],
        weights: dict[str, float],
        calibrator: ProbabilityCalibrator,
        feature_columns: list[str],
    ) -> None:
        self.members = members
        self.weights = weights
        self.calibrator = calibrator
        self.feature_columns = feature_columns

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        train_idx,
        *,
        quick: bool = False,
    ) -> tuple[ReserveEnsemble, dict[str, Any]]:
        """Fit members + weights + calibrator. Returns (ensemble, diagnostics)."""
        member_names = list(get_candidates("classification", quick=quick))
        oof_probabilities: dict[str, np.ndarray] = {}
        diagnostics: dict[str, Any] = {"member_oof_roc_auc": {}}
        for name in member_names:
            factory = get_candidates("classification", quick=quick)[name]
            oof = _grouped_oof_probabilities(factory, X, y, groups, train_idx)
            oof_probabilities[name] = oof
            diagnostics["member_oof_roc_auc"][name] = float(roc_auc_score(y.iloc[train_idx], oof))
        weights = compute_member_weights(oof_probabilities, y.iloc[train_idx])

        members = {
            name: get_candidates("classification", quick=quick)[name]()
            for name in member_names
        }
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        for name, model in members.items():
            model.fit(X_train, y_train)

        raw_oof_ensemble = sum(
            weights[name] * oof_probabilities[name] for name in member_names
        )
        calibrator = fit_calibrator(raw_oof_ensemble, y_train)
        feature_columns = list(X.columns)
        ensemble = cls(members, weights, calibrator, feature_columns)
        diagnostics["weights"] = weights
        diagnostics["calibration_method"] = calibrator.kind
        return ensemble, diagnostics

    def raw_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        total = sum(
            self.weights[name] * model.predict_proba(aligned)[:, 1]
            for name, model in self.members.items()
        )
        return np.asarray(total, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated ensemble probabilities (the serving output)."""
        return self.calibrator.transform(self.raw_probabilities(X))

    def base_probabilities(self, X: pd.DataFrame) -> dict[str, float]:
        """Per-member probabilities for a SINGLE row (API transparency)."""
        aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        return {
            name: float(model.predict_proba(aligned)[0, 1])
            for name, model in self.members.items()
        }

    # ------------------------------------------------------- persistence --

    def save(self, model_path: Path) -> None:
        joblib.dump(
            {
                "members": self.members,
                "weights": self.weights,
                "calibrator": self.calibrator.to_dict(),
                "feature_columns": self.feature_columns,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path: Path) -> ReserveEnsemble:
        payload = joblib.load(model_path)
        return cls(
            members=payload["members"],
            weights=payload["weights"],
            calibrator=ProbabilityCalibrator.from_dict(dict(payload["calibrator"])),
            feature_columns=list(payload["feature_columns"]),
        )


# ----------------------------------------------------------------- training --


def train_prospectivity_ensemble(root: Path | None = None, quick: bool = False) -> dict:
    """Train the calibrated ensemble and compare it with the best baseline.

    Validation protocol (leakage-safe):

    - One isolated spatial split separates the DEVELOPMENT set from the final
      TEST set (``ml.reserve.spatial.spatial_dev_test_split``).
    - Member models, ensemble weights and the monotone calibrator are all fit
      from grouped-CV OOF predictions over the DEVELOPMENT set only.
    - The ensemble-vs-baseline decision is made on development OOF predictions.
    - The untouched spatial TEST set is used exactly once: for the final
      calibrated metrics (ROC-AUC/PR-AUC/F1/Brier/reliability).

    The ensemble is recorded as a candidate model; it is NOT auto-promoted
    (see registry lifecycle).
    """
    from ml.common.registry import ModelVersionRecord, hash_schema, write_registry_record
    from ml.reserve.train_prospectivity import (
        RANDOM_STATE,
        _artifact_dirs,
        next_version,
        training_data_hash,
    )

    root = root or project_root()
    spec = RESERVE_TASKS["prospectivity"]
    model_dir, reserve_dir, registry_dir = _artifact_dirs(root)

    df = load_fused_reserve_table(root)
    X = prepare_reserve_matrix(
        df,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    y = df[spec.target].astype(int)

    assert_no_target_leakage(X, spec.forbidden_columns, context="prospectivity_ensemble")
    from ml.common.validation import check_leakage

    leakage_report = check_leakage(X, [spec.target], context="reserve/prospectivity_ensemble")
    leakage_report.raise_for_errors()

    dev_idx, test_idx, groups = spatial_dev_test_split(df)
    ensemble, diagnostics = ReserveEnsemble.fit(X, y, groups, dev_idx, quick=quick)

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    raw_test = ensemble.raw_probabilities(X_test)
    calibrated_test = ensemble.calibrator.transform(raw_test)

    # FINAL TEST metrics: computed exactly once, on the untouched test rows.
    metrics = classification_metrics(y_test, calibrated_test)
    metrics["brier_raw"] = brier_score(y_test, raw_test)
    metrics["brier_calibrated"] = brier_score(y_test, calibrated_test)
    metrics["raw_reliability"] = reliability_curve(y_test, raw_test)
    metrics["calibrated_reliability"] = reliability_curve(y_test, calibrated_test)
    metrics["final_test_sample_count"] = int(len(test_idx))

    # Development-only baseline comparison.
    # The "does the ensemble beat the best single baseline" decision is made on
    # group-CV OOF predictions over the DEVELOPMENT set, so the untouched final
    # test set never influences it.
    y_dev = y.iloc[dev_idx].reset_index(drop=True)
    baseline_oof: dict[str, np.ndarray] = {}
    for name, factory in get_candidates("classification", quick=quick).items():
        baseline_oof[name] = _grouped_oof_probabilities(factory, X, y, groups, dev_idx)

    dev_ensemble_raw = sum(
        diagnostics["weights"][name] * baseline_oof[name] for name in diagnostics["weights"]
    )
    dev_roc = float(roc_auc_score(y_dev, dev_ensemble_raw))

    baseline_results: dict[str, dict[str, Any]] = {}
    for name, oof in baseline_oof.items():
        baseline_metrics = classification_metrics(y_dev, oof)
        baseline_metrics["brier"] = brier_score(y_dev, oof)
        baseline_results[name] = baseline_metrics
    best_baseline = max(baseline_results, key=lambda name: baseline_results[name]["roc_auc"])

    metrics["development_metrics"] = {
        "protocol": "GroupKFold OOF predictions over development set only",
        "ensemble_oof_roc_auc": dev_roc,
        "member_oof_roc_auc": diagnostics["member_oof_roc_auc"],
        "ensemble_weight_selection_method": (
            "weights proportional to max(OOF ROC-AUC - 0.5, eps) over development set only"
        ),
        "calibration": {
            "method": f"monotone {ensemble.calibrator.kind} fitted on development OOF probabilities",
            "fitted_on": "development_set_oof",
            "development_brier_after_calibration": float(
                brier_score(y_dev, ensemble.calibrator.transform(dev_ensemble_raw))
            ),
        },
    }

    metrics["ensemble_vs_baseline"] = {
        "protocol": "development set GroupKFold OOF comparison (final test untouched)",
        "best_baseline": best_baseline,
        "best_baseline_metrics": baseline_results[best_baseline],
        "ensemble_development_oof_roc_auc": dev_roc,
        "ensemble_preferred": bool(dev_roc >= baseline_results[best_baseline]["roc_auc"]),
        "note": (
            "The ensemble is only preferred when its development OOF ROC-AUC is at "
            "least as good as the best single baseline. This decision uses the "
            "development set ONLY; the final spatial test set is reserved for the "
            "final metrics reported above."
        ),
    }
    metrics.update(
        {
            "validation": "spatial_block_holdout",
            "validation_details": {
                "strategy": "Spatial block holdout with ISOLATED FINAL TEST (GroupShuffleSplit over 5x5 blocks)",
                "primary": True,
                "held_out_blocks": int(groups.iloc[test_idx].nunique()),
                "development_rows": int(len(dev_idx)),
                "final_test_rows": int(len(test_idx)),
                "train_rows": int(len(dev_idx)),
                "test_rows": int(len(test_idx)),
                "random_state": RANDOM_STATE,
            },
            "target": spec.target,
            "leakage_exclusions": spec.forbidden_columns,
            "leakage_check_passed": bool(leakage_report.valid),
            "feature_names": list(X.columns),
            "training_rows": int(len(df)),
            "final_test_sample_count": int(len(test_idx)),
            "final_evaluation_isolation": True,
            "synthetic_data": True,
        }
    )
    metrics.update(group_split_report(groups, dev_idx, test_idx))

    # Support statistics for inference-time data-support assessment.
    # Fitted on the DEVELOPMENT set only — extrapolation reference stats must
    # not learn any final-test information.
    assessor = SupportAssessor.fit(
        X.iloc[dev_idx], df.iloc[dev_idx]["latitude"], df.iloc[dev_idx]["longitude"]
    )

    # Persist (versioned + latest serving copies). No native-XGBoost file —
    # the ensemble bundle is a joblib payload with member models inside.
    version = next_version("reserve_prospectivity_ensemble", registry_dir)
    versioned_dir = reserve_dir / "versions" / version
    versioned_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(versioned_dir / "prospectivity_ensemble_model.joblib")
    joblib.dump(list(X.columns), versioned_dir / "prospectivity_ensemble_features.pkl")
    ensemble.save(reserve_dir / "prospectivity_ensemble_model.joblib")
    joblib.dump(list(X.columns), reserve_dir / "prospectivity_ensemble_features.pkl")

    meta = {
        "model_name": "reserve_prospectivity_ensemble",
        "version": version,
        "task": "binary_classification",
        "task_kind": "classification",
        "algorithm": "WeightedSoftVotingEnsemble+Calibration",
        "target": spec.target,
        "prediction_type": "calibrated_manganese_occurrence_probability",
        "members": list(ensemble.members.keys()),
        "member_weights": ensemble.weights,
        "calibrator": ensemble.calibrator.to_dict(),
        "feature_names": list(X.columns),
        "feature_schema_hash": hash_schema({"features": list(X.columns)})[:16],
        "training_data_hash": training_data_hash(root),
        "validation_strategy": "spatial_block_holdout_with_isolated_final_test",
        "metrics": metrics,
        "development_sample_count": metrics.get("development_sample_count"),
        "final_test_sample_count": metrics.get("final_test_sample_count"),
        "development_spatial_groups": metrics.get("development_spatial_groups"),
        "final_test_spatial_groups": metrics.get("final_test_spatial_groups"),
        "leakage_check_passed": metrics.get("leakage_check_passed"),
        "ensemble_weight_selection_method": (
            "GroupKFold OOF ROC-AUC on development set only (final test untouched)"
        ),
        "calibration_method": f"{ensemble.calibrator.kind} fitted on development OOF probabilities",
        "conformal_calibration_method": None,
        "final_evaluation_isolation": metrics.get("final_evaluation_isolation"),
        "random_seed": RANDOM_STATE,
        "quick_mode": quick,
        "support_stats": assessor.to_dict(),
        "synthetic_data": True,
        "boundary_notice": "PROTOTYPE resource intelligence on synthetic demo data. NOT official mineral reserves; NOT field-validated.",
        "status": "candidate",
    }
    for meta_path in (
        versioned_dir / "prospectivity_ensemble_meta.json",
        reserve_dir / "prospectivity_ensemble_meta.json",
    ):
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    record = ModelVersionRecord(
        model_name="reserve_prospectivity_ensemble",
        version=version,
        task="binary_classification",
        algorithm="WeightedSoftVotingEnsemble+Calibration",
        training_data_hash=meta["training_data_hash"],
        feature_schema_hash=meta["feature_schema_hash"],
        metrics=metrics,
        artifact_path=f"models/reserve/versions/{version}/prospectivity_ensemble_model.joblib",
        status="candidate",
        target=spec.target,
        feature_names=list(X.columns),
        validation_strategy=metrics["validation"],
    )
    write_registry_record(record, registry_dir)

    artifacts = save_evaluation_artifacts(
        model_dir / "reserve" / "evaluation",
        "reserve_prospectivity_ensemble",
        version,
        raw_curve=metrics["raw_reliability"],
        calibrated_curve=metrics["calibrated_reliability"],
        method=ensemble.calibrator.kind,
        comparison={
            "ensemble": {
                k: metrics[k]
                for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier_raw", "brier_calibrated")
            },
            "best_baseline": best_baseline,
            "best_baseline_metrics": baseline_results[best_baseline],
        },
    )

    return {
        "model_name": "reserve_prospectivity_ensemble",
        "version": version,
        "weights": ensemble.weights,
        "calibration_method": ensemble.calibrator.kind,
        "metrics": metrics,
        "artifact_path": record.artifact_path,
        "evaluation_artifacts": artifacts,
    }



