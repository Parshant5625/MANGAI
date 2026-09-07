"""Split-conformal prediction intervals for reserve regression targets.

Methodology (documented, defensible, leakage-aware)
---------------------------------------------------
1. During training, the TRAINING spatial blocks are split with ``GroupKFold``
   (the same grouped protocol as the primary spatial validation) and the
   selected model produces out-of-fold (OOF) predictions for every training
   row. OOF residuals ``|y - pred|`` therefore never involve the spatial
   holdout used for final evaluation.
2. The conformal quantile is the level-``coverage`` empirical quantile of the
   OOF residuals (finite-sample correction ``ceil((n+1)*coverage)/n``).
3. An interval for a new point prediction is ``[pred - q, pred + q]`` — the
   standard symmetric split-conformal interval.
4. Empirical coverage is REPORTED on the untouched spatial holdout. It is an
   empirical property of this synthetic dataset under the grouped-exchangeability
   assumption, not a guarantee for real field data.

Non-negative targets (ore thickness)
------------------------------------
Clipping the lower bound at zero can only RAISE the lower bound, so it can
only widen-or-keep coverage (empirical coverage on the holdout is computed
after clipping and reported). The consequence is documented: intervals near
zero are one-sided in practice.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def conformal_quantile(residuals, coverage: float) -> float:
    """Finite-sample conformal quantile of absolute residuals."""
    residuals = np.abs(np.asarray(residuals, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        raise ValueError("conformal_quantile requires at least one residual")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    n = len(residuals)
    rank = int(np.ceil((n + 1) * coverage))
    rank = min(max(rank, 1), n)
    return float(np.sort(residuals)[rank - 1])


class SplitConformalRegressor:
    """Symmetric split-conformal interval model persisted as plain JSON."""

    def __init__(
        self,
        quantile: float,
        coverage: float,
        n_calibration: int,
        *,
        non_negative: bool = False,
        empirical_coverage: float | None = None,
    ) -> None:
        self.quantile = float(quantile)
        self.coverage = float(coverage)
        self.n_calibration = int(n_calibration)
        self.non_negative = bool(non_negative)
        self.empirical_coverage = empirical_coverage

    @classmethod
    def fit(
        cls,
        oof_predictions,
        oof_truth,
        coverage: float = 0.9,
        *,
        non_negative: bool = False,
    ) -> SplitConformalRegressor:
        residuals = np.asarray(oof_truth, dtype=float) - np.asarray(oof_predictions, dtype=float)
        return cls(
            quantile=conformal_quantile(residuals, coverage),
            coverage=coverage,
            n_calibration=int(len(residuals)),
            non_negative=non_negative,
        )

    def interval(self, prediction: float) -> dict[str, float]:
        """Interval for a single point prediction: lower <= point <= upper.

        For non-negative targets the lower bound is clipped at zero; because
        clipping only raises the lower bound, the reported empirical coverage
        (computed after clipping) is conservative.
        """
        lower = float(prediction - self.quantile)
        upper = float(prediction + self.quantile)
        if self.non_negative and lower < 0.0:
            lower = 0.0
        return {
            "lower": lower,
            "upper": upper,
            "level": self.coverage,
            "method": "split_conformal",
        }

    def evaluate_coverage(self, predictions, truth) -> float:
        """Empirical coverage on an evaluation set (interval contains truth)."""
        predictions = np.asarray(predictions, dtype=float)
        truth = np.asarray(truth, dtype=float)
        covered = 0
        for point, actual in zip(predictions, truth, strict=False):
            interval = self.interval(float(point))
            covered += int(interval["lower"] <= actual <= interval["upper"])
        self.empirical_coverage = float(covered / max(len(truth), 1))
        return self.empirical_coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "split_conformal",
            "quantile": self.quantile,
            "coverage": self.coverage,
            "n_calibration": self.n_calibration,
            "non_negative": self.non_negative,
            "empirical_coverage": self.empirical_coverage,
            "note": (
                "Coverage is empirical on the synthetic spatial holdout and depends "
                "on grouped-exchangeability; not a guarantee for real field data."
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitConformalRegressor:
        return cls(
            quantile=payload["quantile"],
            coverage=payload["coverage"],
            n_calibration=payload["n_calibration"],
            non_negative=payload.get("non_negative", False),
            empirical_coverage=payload.get("empirical_coverage"),
        )
