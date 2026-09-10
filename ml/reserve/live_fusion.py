"""Live reserve feature fusion for real satellite observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.common.provenance import DataBatch

EARTH_RADIUS_M = 6_371_000.0
GEOLOGY_CONTEXT_COLUMNS = (
    "sample_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "depth_m",
    "formation",
)


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


def _validate_geology_context(frame: pd.DataFrame) -> None:
    missing = [column for column in GEOLOGY_CONTEXT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Live geological context is missing required columns: {missing}")
    if frame.empty:
        return
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Live geological context sample_id values must be unique.")
    if not frame["latitude"].between(-90, 90).all() or not frame["longitude"].between(-180, 180).all():
        raise ValueError("Live geological context contains invalid coordinates.")
    if not frame["slope_deg"].between(0, 90).all() or not frame["aspect_deg"].between(0, 360).all():
        raise ValueError("Live geological context contains invalid terrain values.")
    if (frame["depth_m"] < 0).any():
        raise ValueError("Live geological context contains negative depth values.")


def fuse_live_satellite_with_geology(
    satellite_batch: DataBatch,
    geological: pd.DataFrame,
    *,
    max_distance_m: float = 500.0,
) -> LiveReserveFusionResult:
    """Join live satellite pixels to nearest geological context by WGS84 distance.

    Assay and ore-thickness targets are intentionally not required or copied.
    This prevents target leakage in serving-time reserve inference.
    """
    if satellite_batch.provenance.mode != "live":
        raise ValueError("Live reserve fusion requires a live satellite batch.")
    if satellite_batch.provenance.is_synthetic:
        raise ValueError("Synthetic satellite data cannot enter the live reserve path.")
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")

    _validate_geology_context(geological)
    satellite = pd.DataFrame(satellite_batch.records)
    required_satellite = [
        "sample_id", "latitude", "longitude", "blue_b2", "green_b3", "red_b4",
        "nir_b8", "swir_b11", "swir_b12", "land_surface_temperature",
    ]
    missing = [column for column in required_satellite if column not in satellite.columns]
    if missing:
        raise ValueError(f"Satellite batch is missing required reserve features: {missing}")
    if satellite.empty or geological.empty:
        return LiveReserveFusionResult(
            satellite.iloc[0:0].copy(), 0, len(satellite), max_distance_m, satellite_batch.provenance.to_dict()
        )

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
    for column in GEOLOGY_CONTEXT_COLUMNS[1:]:
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
