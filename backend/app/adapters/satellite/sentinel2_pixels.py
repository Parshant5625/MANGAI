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
    from rasterio.warp import reproject
except ImportError:  # pragma: no cover - exercised through the explicit runtime error
    rasterio = None


REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


@dataclass(frozen=True)
class PixelExtractionConfig:
    """Configuration for deterministic Sentinel-2 spectral extraction."""

    target_resolution_m: int = 20
    cloud_mask_keys: tuple[str, ...] = ("SCL_20M", "SCL")
    timeout_seconds: float = 60.0


class Sentinel2PixelExtractor:
    """Read real Sentinel-2 L2A raster assets and derive canonical features.

    The extractor expects asset URLs discovered by the STAC provider. It never
    creates spectral values when an asset is missing and never uses synthetic
    fallback data in live mode.
    """

    def __init__(self, config: PixelExtractionConfig | None = None) -> None:
        self.config = config or PixelExtractionConfig()
        if rasterio is None:
            raise DataUnavailableError(
                "Live Sentinel-2 pixel ingestion requires rasterio to be installed.",
                details={"dependency": "rasterio"},
            )

    def extract_scene(self, scene: dict[str, Any], site_id: str) -> DataBatch:
        assets = scene.get("assets") or {}
        selected = self._select_assets(assets)
        arrays, profile = self._read_and_align(selected)
        mask = self._build_valid_mask(arrays, selected)
        rows = self._features_from_arrays(arrays, mask, profile, site_id, scene.get("scene_id"))
        if not rows:
            raise DataUnavailableError(
                "Sentinel-2 scene contains no valid pixels after quality masking.",
                details={"scene_id": scene.get("scene_id")},
            )

        acquired_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(
            json.dumps(
                {
                    "scene_id": scene.get("scene_id"),
                    "rows": rows,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        valid_ratio = float(mask.mean()) if mask.size else 0.0
        provenance = DataProvenance(
            source_name="Copernicus Data Space Ecosystem Sentinel-2 L2A",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=scene.get("datetime") or acquired_at,
            ingested_at=acquired_at,
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
                    normalized.setdefault(band, href)
        missing = [band for band in REQUIRED_BANDS if band not in normalized]
        if missing:
            raise DataUnavailableError(
                "Sentinel-2 scene is missing required spectral assets.",
                details={"missing_bands": missing},
            )
        return normalized

    def _download(self, href: str) -> bytes:
        request = Request(href, headers={"Accept": "image/tiff, application/octet-stream", "User-Agent": "MANGAI/1.0"})
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError(
                "Sentinel-2 raster asset is unavailable.",
                details={"reason": str(exc)},
            ) from exc

    def _read(self, href: str) -> tuple[np.ndarray, Any]:
        raw = self._download(href)
        try:
            with MemoryFile(BytesIO(raw)) as memory_file:
                with memory_file.open() as dataset:
                    return dataset.read(1).astype(np.float32), dataset.profile.copy()
        except Exception as exc:  # rasterio can raise multiple driver-specific exceptions
            raise DataUnavailableError(
                "Sentinel-2 raster asset could not be decoded.",
                details={"reason": str(exc)},
            ) from exc

    def _read_and_align(self, assets: dict[str, str]) -> tuple[dict[str, np.ndarray], Any]:
        # Use B02 (10 m) as the reference only when target_resolution_m is 10.
        # For the canonical 20 m output, B11 provides the target grid and all
        # 10 m bands are downsampled to it. This avoids inventing sub-20 m detail
        # for the 20 m SWIR bands.
        reference_band = "B11" if self.config.target_resolution_m == 20 else "B02"
        reference, profile = self._read(assets[reference_band])
        profile["width"] = reference.shape[1]
        profile["height"] = reference.shape[0]
        arrays = {reference_band: reference}
        for band, href in assets.items():
            if band == reference_band:
                continue
            array, source_profile = self._read(href)
            destination = np.full(reference.shape, np.nan, dtype=np.float32)
            reproject(
                source=array,
                destination=destination,
                src_transform=source_profile["transform"],
                src_crs=source_profile["crs"],
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
            arrays[band] = destination
        return arrays, profile

    def _build_valid_mask(self, arrays: dict[str, np.ndarray], assets: dict[str, str]) -> np.ndarray:
        valid = np.ones_like(arrays["B11"], dtype=bool)
        for band in REQUIRED_BANDS:
            valid &= np.isfinite(arrays[band])
            valid &= arrays[band] >= 0
            valid &= arrays[band] <= 10000
        # SCL is optional because scene discovery does not guarantee its asset
        # key. If available, mask cloud/shadow/snow classes conservatively.
        for key in self.config.cloud_mask_keys:
            if key in assets:
                scl, _ = self._read(assets[key])
                valid &= ~np.isin(scl, [3, 8, 9, 10, 11])
                break
        return valid

    @staticmethod
    def _features_from_arrays(
        arrays: dict[str, np.ndarray],
        mask: np.ndarray,
        profile: dict[str, Any],
        site_id: str,
        scene_id: str | None,
    ) -> list[dict[str, Any]]:
        b2, b3, b4 = (arrays[band] / 10000.0 for band in ("B02", "B03", "B04"))
        b8, b11, b12 = (arrays[band] / 10000.0 for band in ("B08", "B11", "B12"))
        eps = 1e-6
        ndvi = (b8 - b4) / (b8 + b4 + eps)
        ndwi = (b3 - b8) / (b3 + b8 + eps)
        swir_ratio = b11 / (b12 + eps)
        bare_soil = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)

        rows: list[dict[str, Any]] = []
        transform = profile["transform"]
        crs = profile.get("crs")
        height, width = mask.shape
        for row_idx, col_idx in zip(*np.where(mask)):
            x, y = transform * (int(col_idx) + 0.5, int(row_idx) + 0.5)
            record = {
                "site_id": site_id,
                "sample_id": f"{scene_id or 'scene'}_{row_idx}_{col_idx}",
                "latitude": None,
                "longitude": None,
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
                "x": float(x),
                "y": float(y),
                "crs": str(crs) if crs else None,
            }
            rows.append(record)
        return rows
