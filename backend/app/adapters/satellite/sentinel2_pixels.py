from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from backend.app.core.errors import DataUnavailableError
from ml.common.provenance import DataBatch, DataProvenance

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    from rasterio.warp import reproject, transform, transform_bounds
except ImportError:  # pragma: no cover - exercised through the explicit runtime error
    rasterio = None
    Affine = Any


REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
SCL_MASK_CLASSES = (3, 8, 9, 10, 11)


@dataclass(frozen=True)
class PixelExtractionConfig:
    """Configuration for deterministic Sentinel-2 spectral extraction."""

    target_resolution_m: int = 20
    cloud_mask_keys: tuple[str, ...] = ("SCL_20M", "SCL")
    timeout_seconds: float = 60.0
    aoi_bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.target_resolution_m not in (10, 20):
            raise ValueError("target_resolution_m must be either 10 or 20 metres")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.aoi_bbox is not None:
            min_lon, min_lat, max_lon, max_lat = self.aoi_bbox
            if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
                raise ValueError("aoi_bbox must be WGS84 (min_lon, min_lat, max_lon, max_lat)")


class Sentinel2PixelExtractor:
    """Download, align, mask and extract real Sentinel-2 pixel features."""

    def __init__(self, config: PixelExtractionConfig | None = None) -> None:
        self.config = config or PixelExtractionConfig()

    @staticmethod
    def _require_rasterio() -> None:
        if rasterio is None:
            raise DataUnavailableError("rasterio is required for Sentinel-2 pixel ingestion.")

    @staticmethod
    def _select_assets(scene: dict[str, Any]) -> dict[str, str]:
        assets = scene.get("assets") or {}
        selected: dict[str, str] = {}
        for band in REQUIRED_BANDS:
            for key, asset in assets.items():
                href = asset.get("href") if isinstance(asset, dict) else asset
                if href and key.upper() in {band, f"{band}_20M", f"{band}_10M"}:
                    selected[band] = str(href)
                    break
        missing = [band for band in REQUIRED_BANDS if band not in selected]
        if missing:
            raise DataUnavailableError("Sentinel-2 scene is missing required band assets.", details={"missing_bands": missing})
        return selected

    @staticmethod
    def _select_scl(scene: dict[str, Any]) -> str | None:
        assets = scene.get("assets") or {}
        for key in ("SCL_20M", "SCL_60M", "SCL"):
            asset = assets.get(key)
            href = asset.get("href") if isinstance(asset, dict) else asset
            if href:
                return str(href)
        return None

    def _download(self, href: str) -> bytes:
        try:
            request = Request(href, headers={"User-Agent": "MANGAI/1.0"})
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError("Sentinel-2 raster asset could not be downloaded.", details={"href": href}) from exc

    def _read(self, payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
        self._require_rasterio()
        try:
            with MemoryFile(payload) as memory_file:
                with memory_file.open() as dataset:
                    return dataset.read(1), dataset.profile.copy()
        except Exception as exc:
            raise DataUnavailableError("Sentinel-2 raster asset could not be decoded.") from exc

    def _read_and_align(self, assets: dict[str, str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self._require_rasterio()
        reference_band = "B11" if self.config.target_resolution_m == 20 else "B02"
        reference_array, reference_profile = self._read(self._download(assets[reference_band]))
        if reference_profile.get("crs") is None:
            raise DataUnavailableError("Sentinel-2 raster is missing CRS information.")
        arrays: dict[str, np.ndarray] = {reference_band: reference_array}
        for band, href in assets.items():
            if band == reference_band:
                continue
            source, source_profile = self._read(self._download(href))
            if source_profile.get("crs") is None:
                raise DataUnavailableError("Sentinel-2 raster is missing CRS information.")
            destination = np.empty(reference_array.shape, dtype=np.float32)
            reproject(
                source,
                destination,
                src_transform=source_profile["transform"],
                src_crs=source_profile["crs"],
                dst_transform=reference_profile["transform"],
                dst_crs=reference_profile["crs"],
                resampling=Resampling.bilinear,
            )
            arrays[band] = destination
        return arrays, reference_profile

    @staticmethod
    def _build_valid_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
        valid = np.ones_like(arrays["B11"], dtype=bool)
        for band in REQUIRED_BANDS:
            valid &= np.isfinite(arrays[band])
            valid &= arrays[band] >= 0
            valid &= arrays[band] <= 10000
        return valid

    @staticmethod
    def _features_from_arrays(
        arrays: dict[str, np.ndarray],
        mask: np.ndarray,
        profile: dict[str, Any],
        site_id: str,
        scene_id: str | None,
    ) -> list[dict[str, Any]]:
        crs = profile.get("crs")
        if crs is None:
            raise DataUnavailableError("Sentinel-2 raster is missing CRS information.")
        transform_affine = profile.get("transform")
        if not isinstance(transform_affine, Affine):
            try:
                transform_affine = Affine(*transform_affine)
            except (TypeError, ValueError) as exc:
                raise DataUnavailableError("Sentinel-2 raster has an invalid affine transform.") from exc

        b2, b3, b4 = (arrays[band] / 10000.0 for band in ("B02", "B03", "B04"))
        b8, b11, b12 = (arrays[band] / 10000.0 for band in ("B08", "B11", "B12"))
        eps = 1e-6
        ndvi = (b8 - b4) / (b8 + b4 + eps)
        ndwi = (b3 - b8) / (b3 + b8 + eps)
        swir_ratio = b11 / (b12 + eps)
        bare_soil = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)

        rows_idx, cols_idx = np.where(mask)
        xs: list[float] = []
        ys: list[float] = []
        for row_idx, col_idx in zip(rows_idx, cols_idx):
            x, y = transform_affine * (int(col_idx) + 0.5, int(row_idx) + 0.5)
            xs.append(float(x))
            ys.append(float(y))
        try:
            longitudes, latitudes = transform(crs, "EPSG:4326", xs, ys)
        except Exception as exc:
            raise DataUnavailableError("Sentinel-2 raster coordinates could not be transformed to WGS84.") from exc

        rows: list[dict[str, Any]] = []
        for idx, (row_idx, col_idx) in enumerate(zip(rows_idx, cols_idx)):
            rows.append(
                {
                    "site_id": site_id,
                    "sample_id": f"{scene_id or 'scene'}_{row_idx}_{col_idx}",
                    "latitude": float(latitudes[idx]),
                    "longitude": float(longitudes[idx]),
                    "blue_b2": float(b2[row_idx, col_idx]),
                    "green_b3": float(b3[row_idx, col_idx]),
                    "red_b4": float(b4[row_idx, col_idx]),
                    "nir_b8": float(b8[row_idx, col_idx]),
                    "swir_b11": float(b11[row_idx, col_idx]),
                    "swir_b12": float(b12[row_idx, col_idx]),
                    "ndvi": float(ndvi[row_idx, col_idx]),
                    "ndwi": float(ndwi[row_idx, col_idx]),
                    "swir_ratio": float(swir_ratio[row_idx, col_idx]),
                    "bare_soil_index": float(bare_soil[row_idx, col_idx]),
                    "land_surface_temperature": None,
                    "x": float(xs[idx]),
                    "y": float(ys[idx]),
                    "crs": str(crs),
                }
            )
        return rows
