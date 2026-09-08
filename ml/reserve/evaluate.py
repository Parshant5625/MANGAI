"""Baseline model comparison for the three reserve prediction tasks.

Candidates are deliberately simple, CPU-friendly reference models:

- Classification: Logistic Regression (scaled), Random Forest, XGBoost
- Regression:     Ridge (scaled), Random Forest, XGBoost

Model selection is performed via **grouped cross-validation on the development
set only**. The final untouched spatial test set is never used for candidate
scoring, selection, or any fitting. These are synthetic-data baselines —
never field-validated performance estimates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ml.reserve.spatial import (
    assert_no_group_overlap,
    classification_metrics,
    grouped_cv_scores,
    regression_metrics,
)

RANDOM_STATE = 42
N_JOBS = 4  # fixed thread count keeps training reproducible on laptops


def _scaled(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def classification_candidates() -> dict[str, Callable[[], Any]]:
    return {
        "logistic_regression": lambda: _scaled(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=280,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            tree_method="hist",
        ),
    }


def regression_candidates() -> dict[str, Callable[[], Any]]:
    return {
        "ridge": lambda: _scaled(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "xgboost": lambda: XGBRegressor(
            n_estimators=280,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            tree_method="hist",
        ),
    }


def get_candidates(task_kind: str, quick: bool = False) -> dict[str, Callable[[], Any]]:
    candidates = classification_candidates() if task_kind == "classification" else regression_candidates()
    if quick:
        # Cheap configurations used by the automated test-suite only.
        if task_kind == "classification":
            candidates["random_forest"] = lambda: RandomForestClassifier(
                n_estimators=40, max_depth=8, random_state=RANDOM_STATE, n_jobs=N_JOBS
            )
            candidates["xgboost"] = lambda: XGBClassifier(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",
            )
        else:
            candidates["random_forest"] = lambda: RandomForestRegressor(
                n_estimators=40, max_depth=8, random_state=RANDOM_STATE, n_jobs=N_JOBS
            )
            candidates["xgboost"] = lambda: XGBRegressor(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.1,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",
            )
    return candidates


def score_candidates(
    task_kind: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    dev_idx,
    *,
    quick: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate every candidate via grouped CV on the **development** set only.

    The development set (``dev_idx``) is the only data touched here.  No
    row from the final spatial test set participates in candidate scoring,
    which keeps model selection free of optimistic bias.
    """
    candidates = get_candidates(task_kind, quick=quick)

    # Safety: verify the development split has zero group overlap with the
    # remainder (the final test set).  This is a runtime guard against
    # accidental leakage in the split logic.
    all_indices = np.arange(len(X))
    test_complement = np.setdiff1d(all_indices, dev_idx, assume_unique=False)
    assert_no_group_overlap(groups, dev_idx, test_complement)

    X_dev = X.iloc[dev_idx].reset_index(drop=True)
    y_dev = y.iloc[dev_idx].reset_index(drop=True)
    groups_dev = groups.iloc[dev_idx].reset_index(drop=True)

    results: dict[str, dict[str, Any]] = {}
    for name, factory in candidates.items():
        cv = grouped_cv_scores(factory, X_dev, y_dev, groups_dev, task_kind)
        results[name] = dict(cv)
    return results


def select_best(results: dict[str, dict[str, Any]], task_kind: str) -> str:
    """Select the best candidate by the task's primary **CV** metric.

    Classification maximises ``cv_roc_auc``; regression minimises ``cv_rmse``.
    Both metrics are computed on the **development** cross-validation folds,
    never on the final test set.
    """
    if task_kind == "classification":
        return max(results, key=lambda name: results[name]["cv_roc_auc"])
    return min(results, key=lambda name: results[name]["cv_rmse"])


def eval_on_test(
    task_kind: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_idx,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute final evaluation metrics on the untouched spatial test set.

    This function must be called **after** model fitting is complete and
    must be the only place test-index rows are scored.
    """
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    if task_kind == "classification":
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, probabilities, threshold=threshold)
        metrics["threshold"] = float(threshold)
    else:
        metrics = regression_metrics(y_test, model.predict(X_test))
    return metrics
