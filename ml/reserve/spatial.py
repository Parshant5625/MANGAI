from __future__ import annotations

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
