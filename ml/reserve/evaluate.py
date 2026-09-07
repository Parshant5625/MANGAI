"""Baseline model comparison for the three reserve prediction tasks.

Candidates are deliberately simple, CPU-friendly reference models:

- Classification: Logistic Regression (scaled), Random Forest, XGBoost
- Regression:     Ridge (scaled), Random Forest, XGBoost

All candidates are evaluated with the SAME spatial-block validation strategy
(``ml.reserve.spatial``). The winner is selected by the task's primary metric
(ROC-AUC for classification, RMSE for regression), but every candidate's full
metric set is retained for reporting. These are synthetic-data baselines —
never field-validated performance estimates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ml.reserve.spatial import classification_metrics, grouped_cv_scores, regression_metrics

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
    train_idx,
    test_idx,
    *,
    quick: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate every candidate for a task kind on the spatial holdout + CV."""
    candidates = get_candidates(task_kind, quick=quick)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    results: dict[str, dict[str, Any]] = {}
    for name, factory in candidates.items():
        model = factory()
        model.fit(X_train, y_train)
        if task_kind == "classification":
            probabilities = model.predict_proba(X_test)[:, 1]
            holdout = classification_metrics(y_test, probabilities)
        else:
            holdout = regression_metrics(y_test, model.predict(X_test))
        cv = grouped_cv_scores(factory, X, y, groups, task_kind)
        results[name] = {**holdout, **cv}
    return results


def select_best(results: dict[str, dict[str, Any]], task_kind: str) -> str:
    """Select the best candidate by the task's primary metric.

    Classification maximises ``roc_auc``; regression minimises ``rmse``.
    """
    if task_kind == "classification":
        return max(results, key=lambda name: results[name]["roc_auc"])
    return min(results, key=lambda name: results[name]["rmse"])
