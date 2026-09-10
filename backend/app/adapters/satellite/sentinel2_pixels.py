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
except ImportError:  # pragma: no cover
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
            raise DataUnavailableError("Sentinel-2 target resolution must be 10m or 20m.", details={"target_resolution_m": self.target_resolution_m})
        if self.timeout_seconds <= 0:
            raise DataUnavailableError("Sentinel-2 pixel ingestion timeout must be positive.", details={"timeout_seconds": self.timeout_seconds})
        if self.aoi_bbox is not None:
            min_lon, min_lat, max_lon, max_lat = self.aoi_bbox
            if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
                raise DataUnavailableError("Sentinel-2 AOI bbox must be (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.")


class Sentinel2PixelExtractor:
    """Read real Sentinel-2 L2A assets, align bands, clip AOI and mask SCL."""

    def __init__(self, config: PixelExtractionConfig | None = None) -> None:
        self.config = config or PixelExtractionConfig()
        if rasterio is None:
            raise DataUnavailableError("Live Sentinel-2 pixel ingestion requires rasterio to be installed.", details={"dependency": "rasterio"})

    def extract_scene(self, scene: dict[str, Any], site_id: str) -> DataBatch:
        assets = scene.get("assets") or {}
        selected = self._select_assets(assets)
        scl_href = self._select_scl(assets)
        arrays, profile = self._read_and_align(selected)
        mask = self._build_valid_mask(arrays)
        if scl_href:
            scl, _ = self._read_and_align({"SCL": scl_href}, reference_profile=profile)
            scl_values = scl["SCL"]
            mask &= np.isfinite(scl_values)
            mask &= ~np.isin(scl_values.astype(np.int16), SCL_MASK_CLASSES)
        rows = self._features_from_arrays(arrays, mask, profile, site_id, scene.get("scene_id"))
        if not rows:
            raise DataUnavailableError("Sentinel-2 scene contains no valid pixels after AOI and quality masking.", details={"scene_id": scene.get("scene_id")})
        ingested_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(json.dumps({"scene_id": scene.get("scene_id"), "rows": rows}, sort_keys=True).encode("utf-8")).hexdigest()
        valid_ratio = float(mask.mean()) if mask.size else 0.0
        provenance = DataProvenance(
            source_name="Copernicus Data Space Ecosystem Sentinel-2 L2A",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=scene.get("datetime") or ingested_at,
            ingested_at=ingested_at,
            source_version=str(scene.get("collection") or "sentinel-2-l2a"),
            source_uri=str(scene.get("scene_id") or "copernicus-stac"),
            checksum=checksum,
            license_note="Copernicus Sentinel data are made available free of charge; verify applicable access and usage terms for deployment.",
            quality_score=valid_ratio,
            row_count=len(rows),
        )
        return DataBatch(records=rows, provenance=provenance)

    @staticmethod
    def _select_assets(assets: dict[str, Any]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in assets.items():
            href = value if isinstance(value, str) else value.get("href") if isinstance(value, dict) else None
            if not href:
                continue
            upper = key.upper()
            for band in REQUIRED_BANDS:
                if upper == band or upper.startswith(f"{band}_"):
                    normalized.setdefault(band, str(href))
        missing = [band for band in REQUIRED_BANDS if band not in normalized]
        if missing:
            raise DataUnavailableError("Sentinel-2 scene is missing required spectral assets.", details={"missing_bands": missing})
        return normalized

    def _select_scl(self, assets: dict[str, Any]) -> str | None:
        for key in self.config.cloud_mask_keys:
            value = assets.get(key)
            if isinstance(value, dict):
                value = value.get("href")
            if value:
                return str(value)
        return None

    def _download(self, href: str) -> bytes:
        request = Request(href, headers={"Accept": "image/tiff, application/octet-stream", "User-Agent": "MANGAI/1.0"})
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError("Sentinel-2 raster asset is unavailable.", details={"reason": str(exc)}) from exc

    def _read(self, href: str) -> tuple[np.ndarray, Any]:
        raw = self._download(href)
        try:
            with MemoryFile(BytesIO(raw)) as memory_file:
                with memory_file.open() as dataset:
                    return dataset.read(1).astype(np.float32), dataset.profile.copy()
        except Exception as exc:
            raise DataUnavailableError("Sentinel-2 raster asset could not be decoded.", details={"reason": str(exc)}) from exc

    def _read_and_align(self, assets: dict[str, str], reference_profile: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], Any]:
        reference_band = next(iter(assets)) if reference_profile is not None else ("B11" if self.config.target_resolution_m == 20 else "B02")
        reference, source_profile = self._read(assets[reference_band])
        if source_profile.get("crs") is None:
            raise DataUnavailableError("Sentinel-2 raster is missing CRS information.")
        if reference_profile is None:
            profile = source_profile.copy()
            profile["width"] = reference.shape[1]
            profile["height"] = reference.shape[0]
            if self.config.aoi_bbox is not None:
                bounds = transform_bounds("EPSG:4326", profile["crs"], *self.config.aoi_bbox, densify_pts=21)
                left, bottom, right, top = bounds
                col_start = max(0, int(np.floor((left - profile["transform"].c) / profile["transform"].a)))
                col_stop = min(profile["width"], int(np.ceil((right - profile["transform"].c) / profile["transform"].a)))
                row_start = max(0, int(np.floor((profile["transform"].f - top) / abs(profile["transform"].e))))
                row_stop = min(profile["height"], int(np.ceil((profile["transform"].f - bottom) / abs(profile["transform"].e))))
                if col_start >= col_stop or row_start >= row_stop:
                    raise DataUnavailableError("Sentinel-2 AOI does not intersect the raster extent.")
                reference = reference[row_start:row_stop, col_start:col_stop]
                profile["transform"] = profile["transform"] * Affine.translation(col_start, row_start)
                profile["width"] = reference.shape[1]
                profile["height"] = reference.shape[0]
            arrays = {reference_band: reference}
        else:
            profile = reference_profile.copy()
            arrays = {}
        for band, href in assets.items():
            if reference_profile is None and band == reference_band:
                continue
            array, band_profile = self._read(href)
            if band_profile.get("crs") is None:
                raise DataUnavailableError("Sentinel-2 raster is missing CRS information.", details={"band": band})
            destination = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
            reproject(
                source=array,
                destination=destination,
                src_transform=band_profile["transform"],
                src_crs=band_profile["crs"],
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                resampling=Resampling.nearest if band == "SCL" else Resampling.bilinear,
                dst_nodata=np.nan,
            )
            arrays[band] = destination
        return arrays, profile

    @staticmethod
    def _build_valid_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
        valid = np.ones_like(arrays["B11"], dtype=bool)
        for band in REQUIRED_BANDS:
            valid &= np.isfinite(arrays[band])
            valid &= arrays[band] >= 0
            valid &= arrays[band] <= 10000
        return valid

    @staticmethod
    def _features_from_arrays(arrays: dict[str, np.ndarray], mask: np.ndarray, profile: dict[str, Any], site_id: str, scene_id: str | None) -> list[dict[str, Any]]:
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
            rows.append({
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
            })
        return rows
