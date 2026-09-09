"""Probability calibration and evaluation utilities for reserve prospectivity.

Methodology
-----------
Calibration is fitted on OUT-OF-FOLD spatial predictions: the training spatial
blocks are split with ``GroupKFold`` (same protocol as the primary spatial
validation), base models produce out-of-fold probabilities for the training
rows, and a calibrator (isotonic regression or Platt/sigmoid scaling) is fitted
on those out-of-fold pairs. The calibrator therefore never sees the spatial
holdout used for final evaluation, and the final holdout remains untouched by
calibration fitting — no calibration leakage.

Method selection: the calibrator with the LOWER Brier score on the
out-of-fold pairs is selected. Accuracy is deliberately NOT a calibration
metric. ROC-AUC/PR-AUC are preserved by monotone calibrators (both isotonic
and sigmoid are monotone non-decreasing).

All data here is SYNTHETIC — calibration quality claims apply to the synthetic
validation protocol only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Matplotlib is only needed for diagram export; force a headless backend.
import matplotlib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

DEFAULT_BINS = 10


def brier_score(y_true, probabilities) -> float:
    """Mean squared difference between outcome and predicted probability."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        raise ValueError("brier_score requires at least one sample")
    return float(np.mean((p - y) ** 2))


def reliability_curve(y_true, probabilities, n_bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Binned reliability data for a calibration (reliability) diagram.

    Returns per-bin mean predicted probability, observed positive rate and
    sample count, plus the overall Brier score. Consumers (plot or API) use
    the actual predictions — nothing is hard-coded.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("reliability_curve requires aligned non-empty arrays")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        bins.append(
            {
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else None,
                "observed_rate": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "n_bins": n_bins,
        "n_samples": int(len(y)),
        "brier": brier_score(y, p),
        "bins": bins,
    }


class ProbabilityCalibrator:
    """Monotone probability calibrator persisted as plain JSON.

    kind == "isotonic": piecewise-linear isotonic regression mapping.
    kind == "sigmoid":  Platt scaling — logistic regression on the logit of
    the raw probability.
    """

    def __init__(self, kind: str, **state: Any) -> None:
        if kind not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Unknown calibrator kind: {kind}")
        self.kind = kind
        self.state = state

    @classmethod
    def fit_isotonic(cls, probabilities, y_true) -> ProbabilityCalibrator:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(np.asarray(probabilities, dtype=float), np.asarray(y_true, dtype=float))
        return cls(
            "isotonic",
            x_thresholds=model.X_thresholds_.tolist(),
            y_thresholds=model.y_thresholds_.tolist(),
        )

    @classmethod
    def fit_sigmoid(cls, probabilities, y_true) -> ProbabilityCalibrator:
        raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1 - raw)).reshape(-1, 1)
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(logit, np.asarray(y_true, dtype=float).ravel())
        return cls("sigmoid", coef=float(model.coef_[0][0]), intercept=float(model.intercept_[0]))

    def transform(self, probabilities) -> np.ndarray:
        raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if self.kind == "isotonic":
            return np.interp(
                raw,
                np.asarray(self.state["x_thresholds"], dtype=float),
                np.asarray(self.state["y_thresholds"], dtype=float),
            )
        logit = np.log(raw / (1 - raw))
        return 1.0 / (1.0 + np.exp(-(self.state["coef"] * logit + self.state["intercept"])))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.state}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbabilityCalibrator:
        kind = payload.pop("kind")
        return cls(kind, **payload)


def fit_calibrator(probabilities, y_true) -> ProbabilityCalibrator:
    """Select isotonic vs sigmoid by Brier score on the given OOF pairs."""
    candidates = {
        "isotonic": ProbabilityCalibrator.fit_isotonic(probabilities, y_true),
        "sigmoid": ProbabilityCalibrator.fit_sigmoid(probabilities, y_true),
    }
    scored = {
        name: brier_score(y_true, calibrator.transform(probabilities))
        for name, calibrator in candidates.items()
    }
    best = min(scored, key=scored.get)
    return candidates[best]


def save_evaluation_artifacts(
    evaluation_dir: Path,
    model_name: str,
    version: str,
    *,
    raw_curve: dict[str, Any],
    calibrated_curve: dict[str, Any],
    method: str,
    comparison: dict[str, Any],
) -> dict[str, str]:
    """Persist calibration evaluation artifacts (JSON data + reliability PNG).

    Plot functions consume the actual reliability curves — no hard-coded data.
    """
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{version}"
    json_path = evaluation_dir / f"{stem}_calibration.json"
    json_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "version": version,
                "calibration_method": method,
                "raw_reliability": raw_curve,
                "calibrated_reliability": calibrated_curve,
                "comparison": comparison,
                "synthetic_data": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    png_path = evaluation_dir / f"{stem}_reliability_diagram.png"
    _plot_reliability_diagram(raw_curve, calibrated_curve, model_name, version, method, png_path)
    return {"calibration_json": str(json_path), "reliability_png": str(png_path)}


def _plot_reliability_diagram(
    raw_curve: dict[str, Any],
    calibrated_curve: dict[str, Any],
    model_name: str,
    version: str,
    method: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for curve, label, color in (
        (raw_curve, f"raw (Brier={raw_curve['brier']:.4f})", "tab:orange"),
        (calibrated_curve, f"calibrated[{method}] (Brier={calibrated_curve['brier']:.4f})", "tab:blue"),
    ):
        points = [
            (entry["mean_predicted"], entry["observed_rate"])
            for entry in curve["bins"]
            if entry["count"] > 0 and entry["mean_predicted"] is not None
        ]
        if points:
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, marker="o", color=color, label=label)
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed manganese rate (bin)")
    ax.set_title(f"Reliability — {model_name} {version} (synthetic data)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
