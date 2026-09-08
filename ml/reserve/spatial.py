from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from ml.common.metrics import mae, r2, rmse
from ml.common.preprocessing import spatial_block_id


def spatial_holdout_indices(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    blocks: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    groups = spatial_block_id(df, blocks=blocks)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dummy = np.zeros(len(df))
    train_idx, test_idx = next(splitter.split(df, dummy, groups))
    return train_idx, test_idx, groups


def spatial_dev_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    blocks: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Spatial split into a development set and a final untouched test set.

    Returns ``(dev_idx, test_idx, groups)`` where *dev_idx* and *test_idx* share
    **zero** spatial block groups.  The development set is used for all model
    selection, weight selection, calibration, and conformal quantile fitting
    (via grouped cross-validation).  The test set is reserved for final
    evaluation only and is never touched during development.
    """
    dev_idx, test_idx, groups = spatial_holdout_indices(
        df, test_size=test_size, random_state=random_state, blocks=blocks
    )
    assert_no_group_overlap(groups, dev_idx, test_idx)
    return dev_idx, test_idx, groups


def assert_no_group_overlap(
    groups: pd.Series,
    dev_idx,
    test_idx,
) -> None:
    """Raise ``ValueError`` if any spatial group appears in both dev and test."""
    dev_groups = set(groups.iloc[dev_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = dev_groups & test_groups
    if overlap:
        raise ValueError(
            f"Spatial group overlap between development and final test: {sorted(overlap)}"
        )


def group_split_report(groups: pd.Series, dev_idx, test_idx) -> dict[str, Any]:
    """Machine-readable group separation report for metadata and audits.

    Returns the development/test spatial groups, their intersection (must be
    empty) and a ``leakage_check_passed`` boolean. Persisted in model metadata
    so every version carries a programmatic proof of test isolation.
    """
    dev_groups = sorted(str(group) for group in set(groups.iloc[dev_idx]))
    test_groups = sorted(str(group) for group in set(groups.iloc[test_idx]))
    overlap = sorted(set(dev_groups) & set(test_groups))
    return {
        "development_spatial_groups": dev_groups,
        "final_test_spatial_groups": test_groups,
        "group_overlap": overlap,
        "leakage_check_passed": not overlap,
        "development_sample_count": int(len(dev_idx)),
        "final_test_sample_count": int(len(test_idx)),
    }


def random_holdout_indices(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random i.i.d. holdout split — DIAGNOSTIC ONLY.

    Mineralization is spatially clustered, so a random split lets spatially
    adjacent (near-duplicate) samples leak between train and validation and
    typically reports optimistic metrics. It is recorded next to the spatial
    split purely to quantify that optimism; it must never be the primary
    reported validation.
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state
    )
    return train_idx, test_idx


def assert_no_target_leakage(
    X: pd.DataFrame,
    forbidden: list[str] | tuple[str, ...],
    *,
    context: str = "",
) -> None:
    """Raise ValueError if any forbidden (target-derived) column entered X."""
    leaked = sorted(set(X.columns) & set(forbidden))
    if leaked:
        prefix = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Target leakage detected in feature matrix. "
            f"Forbidden columns present: {leaked}"
        )


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mae(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2(y_true, y_pred)),
    }


def grouped_cv_scores(estimator_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series, task: str) -> dict[str, float]:
    unique_groups = groups.nunique()
    splits = min(5, max(2, unique_groups))
    fold = GroupKFold(n_splits=splits)
    scores: list[dict[str, float]] = []
    for train_idx, test_idx in fold.split(X, y, groups):
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        if task == "classification":
            probs = model.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(classification_metrics(y.iloc[test_idx], probs))
        else:
            preds = model.predict(X.iloc[test_idx])
            scores.append(regression_metrics(y.iloc[test_idx], preds))
    # Non-scalar entries (e.g. the confusion matrix) are excluded from averaging.
    keys = [key for key in scores[0] if isinstance(scores[0][key], (int, float))]
    return {f"cv_{key}": float(np.mean([item[key] for item in scores])) for key in keys}
