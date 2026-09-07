"""Data-support and extrapolation indicators for reserve predictions.

Two SEPARATE concepts (deliberately not merged into one "confidence"):

1. MODEL UNCERTAINTY — how uncertain the trained model is about its own
   prediction (calibration reliability, prediction-interval width). Owned by
   the model/calibration modules.

2. DATA SUPPORT — how well the input point is represented by the training
   data (feature-space proximity + observation proximity + completeness).
   A prediction far from any observation can still have a "confident" model
   output while resting on extrapolation; this module surfaces that.

Extrapolation method (lightweight, documented)
----------------------------------------------
Standardized feature distance: each feature is z-scored with the TRAINING
mean/std; the score is the mean absolute z-score over available features.
Thresholds (heuristic, documented, not scientific truth):
    score < 1.0  -> well_supported
    score < 2.5  -> moderate_support
    else         -> extrapolation_warning
This is a WARNING mechanism — it never proves a prediction wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Approximate metres per degree at manganese-belt latitudes (~21°N).
METRES_PER_DEGREE_LAT = 110_574.0
METRES_PER_DEGREE_LON = 103_000.0

SUPPORT_LEVELS = ("well_supported", "moderate_support", "extrapolation_warning")
MODERATE_THRESHOLD = 1.0
WARNING_THRESHOLD = 2.5


def standardized_distance_score(
    features: dict[str, float], means: dict[str, float], stds: dict[str, float]
) -> float:
    """Mean absolute z-score across the features present in both mappings."""
    scores = []
    for name, value in features.items():
        if name not in means or name not in stds:
            continue
        std = stds[name]
        if not std or np.isnan(std) or value is None or np.isnan(value):
            continue
        scores.append(abs(float(value) - means[name]) / std)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def support_level_from_score(score: float) -> str:
    if np.isnan(score):
        return "extrapolation_warning"
    if score < MODERATE_THRESHOLD:
        return "well_supported"
    if score < WARNING_THRESHOLD:
        return "moderate_support"
    return "extrapolation_warning"


class SupportAssessor:
    """Fits training-feature statistics and assesses new points.

    ``fit`` consumes the TRAINING feature matrix only. The fitted state is
    small (per-feature mean/std + a coordinate subsample) so it is persisted
    inside model metadata for inference-time assessment.
    """

    def __init__(
        self,
        means: dict[str, float],
        stds: dict[str, float],
        coordinates: list[dict[str, float]],
        max_features: int,
    ) -> None:
        self.means = means
        self.stds = stds
        self.coordinates = coordinates
        self.max_features = max_features

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        latitudes: pd.Series,
        longitudes: pd.Series,
        *,
        max_coords: int = 250,
        random_state: int = 42,
    ) -> SupportAssessor:
        numeric = X.select_dtypes(include=[np.number])
        means = {name: float(numeric[name].mean()) for name in numeric.columns}
        stds = {name: float(numeric[name].std()) for name in numeric.columns}
        coords = pd.DataFrame({"latitude": latitudes, "longitude": longitudes})
        if len(coords) > max_coords:
            coords = coords.sample(n=max_coords, random_state=random_state)
        coordinates = [
            {"latitude": float(row.latitude), "longitude": float(row.longitude)}
            for row in coords.itertuples(index=False)
        ]
        return cls(means, stds, coordinates, max_features=len(numeric.columns))

    def assess(
        self,
        features: dict[str, float],
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        """Extrapolation level + data-support metrics for one prediction."""
        score = standardized_distance_score(features, self.means, self.stds)
        known = sum(1 for value in features.values() if value is not None and not np.isnan(value))
        completeness = known / max(len(features), 1)
        distance_m = None
        within_radius = None
        if latitude is not None and longitude is not None and self.coordinates:
            squared = [
                (abs(latitude - point["latitude"]) * METRES_PER_DEGREE_LAT) ** 2
                + (abs(longitude - point["longitude"]) * METRES_PER_DEGREE_LON) ** 2
                for point in self.coordinates
            ]
            distance_m = float(min(squared) ** 0.5)
            within_radius = int(sum(1 for d2 in squared if d2 <= 5_000.0**2))
        return {
            "extrapolation": {
                "method": "standardized_feature_distance",
                "score": None if np.isnan(score) else round(score, 4),
                "level": support_level_from_score(score),
                "thresholds": {"moderate": MODERATE_THRESHOLD, "warning": WARNING_THRESHOLD},
                "note": "Warning mechanism only — does not prove a prediction is wrong.",
            },
            "data_support": {
                "feature_completeness": round(completeness, 3),
                "features_expected": len(features),
                "nearest_observation_m": None if distance_m is None else round(distance_m, 1),
                "observations_within_5km": within_radius,
                "reference_observations": len(self.coordinates),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "means": self.means,
            "stds": self.stds,
            "coordinates": self.coordinates,
            "max_features": self.max_features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SupportAssessor:
        return cls(
            means=payload["means"],
            stds=payload["stds"],
            coordinates=payload["coordinates"],
            max_features=payload["max_features"],
        )
