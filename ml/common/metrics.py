from __future__ import annotations

import math
from collections.abc import Iterable


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    pairs = list(zip(y_true, y_pred, strict=False))
    return sum(abs(a - b) for a, b in pairs) / max(len(pairs), 1)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    pairs = list(zip(y_true, y_pred, strict=False))
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / max(len(pairs), 1))


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    truth = list(y_true)
    pred = list(y_pred)
    if not truth:
        return 0.0
    mean_truth = sum(truth) / len(truth)
    ss_res = sum((a - b) ** 2 for a, b in zip(truth, pred, strict=False))
    ss_tot = sum((a - mean_truth) ** 2 for a in truth)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot

