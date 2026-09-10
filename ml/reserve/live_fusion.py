"""Live reserve feature fusion for real satellite observations.

The production reserve path keeps real satellite provenance and joins it to
non-target geological context by spatial proximity. Assay/ore targets are
never used as serving features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.common.contracts import GEOLOGICAL_CONTEXT_CONTRACT
from ml.common.provenance import DataBatch
from ml.common.validation import validate_dataset

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class LiveReserveFusionResult:
    data: pd.DataFrame
    matched: int
    unmatched_satellite: int
    max_distance_m: float
    satellite_provenance: dict

    @property
    def match_rate(self) -> float:
        total = self.matched + self.unmatched_satellite
        return self.matched / total if total else 0.0


def _haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_r = np.radians(lat1)[:, None]
    lon1_r = np.radians(lon1)[:, None]
    lat2_r = np.radians(lat2)[None, :]
    lon2_r = np.radians(lon2)[None, :]
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fuse_live_satellite_with_geology(
    satellite_batch: DataBatch,
    geological: pd.DataFrame,
    *,
    max_distance_m: float = 500.0,
) -> LiveReserveFusionResult:
    """Join live satellite pixels to the nearest geological context observation.

    ``geological`` must contain contextual columns only. Target assay columns
    are deliberately not required, preventing target leakage during inference.
    """
    if satellite_batch.provenance.mode != "live":
        raise ValueError("Live reserve fusion requires a live satellite batch.")
    if satellite_batch.provenance.is_synthetic:
        raise ValueError("Synthetic satellite data cannot enter the live reserve path.")
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")

    validation = validate_dataset(geological, GEOLOGICAL_CONTEXT_CONTRACT)
    validation.raise_for_errors()
    satellite = pd.DataFrame(satellite_batch.records)
    required_satellite = ["sample_id", "latitude", "longitude", "blue_b2", "green_b3", "red_b4", "nir_b8", "swir_b11", "swir_b12", "land_surface_temperature"]
    missing = [column for column in required_satellite if column not in satellite.columns]
    if missing:
        raise ValueError(f"Satellite batch is missing required reserve features: {missing}")
    if satellite.empty or geological.empty:
        return LiveReserveFusionResult(satellite.iloc[0:0].copy(), 0, len(satellite), max_distance_m, satellite_batch.provenance.to_dict())

    distances = _haversine_matrix(
        satellite["latitude"].to_numpy(float),
        satellite["longitude"].to_numpy(float),
        geological["latitude"].to_numpy(float),
        geological["longitude"].to_numpy(float),
    )
    nearest = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(satellite)), nearest]
    matched_mask = nearest_distance <= max_distance_m
    sat = satellite.loc[matched_mask].reset_index(drop=True)
    context = geological.iloc[nearest[matched_mask]].reset_index(drop=True).add_suffix("_geo")
    fused = pd.concat([sat, context], axis=1)
    for column in GEOLOGICAL_CONTEXT_CONTRACT.column_names():
        if column == "sample_id":
            continue
        geo_column = f"{column}_geo"
        if geo_column in fused:
            fused[column] = fused[geo_column]
    fused["geology_match_distance_m"] = nearest_distance[matched_mask]
    fused["satellite_source"] = satellite_batch.provenance.source_name
    fused["satellite_provenance_checksum"] = satellite_batch.provenance.checksum
    fused["satellite_acquired_at"] = satellite_batch.provenance.acquired_at
    return LiveReserveFusionResult(
        fused,
        int(matched_mask.sum()),
        int((~matched_mask).sum()),
        max_distance_m,
        satellite_batch.provenance.to_dict(),
    )
