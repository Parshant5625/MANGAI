from __future__ import annotations

import hashlib
import json
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
    from rasterio.io import MemoryFile
    from rasterio.warp import transform, transform_bounds
    from rasterio.windows import Window, from_bounds
except ImportError:  # pragma: no cover
    rasterio = None

ST_SCALE = 0.00341802
ST_OFFSET_K = 149.0
ST_MIN_C = -50.0
ST_MAX_C = 85.0
THERMAL_NEIGHBOR_RADIUS_PIXELS = 2


class LandsatSurfaceTemperatureFusion:
    """Sample real Landsat Collection 2 ST at Sentinel-2 pixel locations."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if rasterio is None:
            raise DataUnavailableError("Landsat ST fusion requires rasterio to be installed.")
        if timeout_seconds <= 0:
            raise DataUnavailableError("Landsat ST timeout must be positive.")
        self.timeout_seconds = timeout_seconds

    def fuse(self, optical_batch: DataBatch, thermal_scene: dict[str, Any], *, aoi_bbox: tuple[float, float, float, float] | None = None) -> DataBatch:
        if optical_batch.provenance.mode != "live":
            raise DataUnavailableError("Thermal fusion is only enabled for live satellite batches.")
        if optical_batch.provenance.dataset != "satellite_features":
            raise DataUnavailableError("Thermal fusion expects satellite_features records.")
        asset = (thermal_scene.get("assets") or {}).get("surface_temperature")
        if not asset:
            raise DataUnavailableError("Landsat ST scene has no surface temperature asset.")
        if not optical_batch.records:
            raise DataUnavailableError("Cannot fuse thermal data into an empty optical batch.")

        href = str(asset)
        try:
            if aoi_bbox is not None and href.startswith(("http://", "https://")):
                thermal_values = self._sample_remote_cog(href, optical_batch, aoi_bbox)
            else:
                raw = self._download(href)
                with MemoryFile(BytesIO(raw)) as memory_file:
                    with memory_file.open() as dataset:
                        thermal_values = self._sample_dataset(dataset, optical_batch)
        except DataUnavailableError:
            raise
        except Exception as exc:
            raise DataUnavailableError(
                "Landsat ST raster could not be decoded or sampled.",
                details={"reason": f"{type(exc).__name__}: {exc}", "asset": href.split("?")[0]},
            ) from exc

        temperatures_c = thermal_values * ST_SCALE + ST_OFFSET_K - 273.15
        valid = np.isfinite(thermal_values) & (thermal_values > 0) & (thermal_values <= 65535) & np.isfinite(temperatures_c) & (temperatures_c >= ST_MIN_C) & (temperatures_c <= ST_MAX_C)
        if not valid.any():
            raise DataUnavailableError(
                "Landsat ST scene has no valid thermal pixels at Sentinel locations.",
                details={"temperature_range_c": [ST_MIN_C, ST_MAX_C]},
            )

        records: list[dict[str, Any]] = []
        for record, temperature_c, is_valid in zip(optical_batch.records, temperatures_c, valid):
            if not is_valid:
                continue
            updated = dict(record)
            updated["land_surface_temperature"] = float(temperature_c)
            updated["thermal_source"] = "landsat-collection-2-surface-temperature"
            updated["thermal_scene_id"] = thermal_scene.get("scene_id")
            updated["thermal_resampling"] = "nearest-pixel"
            records.append(updated)

        coverage = len(records) / len(optical_batch.records)
        ingested_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        provenance = DataProvenance(
            source_name="Copernicus Sentinel-2 L2A + USGS Landsat Collection 2 ST via Microsoft Planetary Computer",
            source_kind="satellite", mode="live", dataset="satellite_features",
            acquired_at=thermal_scene.get("datetime") or optical_batch.provenance.acquired_at,
            ingested_at=ingested_at,
            source_version=f"{optical_batch.provenance.source_version or 'sentinel-2'}+{thermal_scene.get('collection', 'landsat-c2-l2')}",
            source_uri=str(thermal_scene.get("source_uri") or thermal_scene.get("scene_id") or "microsoft-planetary-computer"),
            checksum=checksum,
            license_note="Sentinel-2 and Landsat source terms must be verified for deployment; USGS Landsat Collection 2 is publicly accessible.",
            quality_score=coverage, row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)

    def _sample_dataset(self, dataset: Any, optical_batch: DataBatch) -> np.ndarray:
        if dataset.crs is None:
            raise DataUnavailableError("Landsat ST raster is missing CRS information.")
        source_crs = optical_batch.records[0].get("crs")
        if not source_crs:
            raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
        xs = [float(record["x"]) for record in optical_batch.records]
        ys = [float(record["y"]) for record in optical_batch.records]
        target_x, target_y = transform(source_crs, dataset.crs, xs, ys)
        return self._sample_points_with_neighborhood(dataset, target_x, target_y)

    def _sample_remote_cog(self, href: str, optical_batch: DataBatch, aoi_bbox: tuple[float, float, float, float]) -> np.ndarray:
        env_options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
            "GDAL_HTTP_VERSION": "1.1", "GDAL_HTTP_MULTIPLEX": "NO", "GDAL_HTTP_MAX_RETRY": "4",
            "GDAL_HTTP_RETRY_DELAY": "1", "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "5000000",
        }
        with rasterio.Env(**env_options):
            with rasterio.open(href, sharing=False) as dataset:
                if dataset.crs is None:
                    raise DataUnavailableError("Landsat ST raster is missing CRS information.")
                source_crs = optical_batch.records[0].get("crs")
                if not source_crs:
                    raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
                left, bottom, right, top = transform_bounds("EPSG:4326", dataset.crs, *aoi_bbox, densify_pts=21)
                requested = from_bounds(left, bottom, right, top, transform=dataset.transform)
                expanded = Window(requested.col_off - THERMAL_NEIGHBOR_RADIUS_PIXELS, requested.row_off - THERMAL_NEIGHBOR_RADIUS_PIXELS, requested.width + 2 * THERMAL_NEIGHBOR_RADIUS_PIXELS, requested.height + 2 * THERMAL_NEIGHBOR_RADIUS_PIXELS)
                window = expanded.intersection(Window(0, 0, dataset.width, dataset.height))
                if window.width <= 0 or window.height <= 0:
                    raise DataUnavailableError("Landsat ST raster does not overlap the requested AOI.")
                array = dataset.read(1, window=window, masked=True)
                xs = [float(record["x"]) for record in optical_batch.records]
                ys = [float(record["y"]) for record in optical_batch.records]
                target_x, target_y = transform(source_crs, dataset.crs, xs, ys)
                values = np.full(len(xs), np.nan, dtype=np.float32)
                for index, (x, y) in enumerate(zip(target_x, target_y)):
                    row, col = rasterio.transform.rowcol(dataset.transform, x, y)
                    local_row = int(row) - int(window.row_off)
                    local_col = int(col) - int(window.col_off)
                    candidate = self._nearest_valid_from_array(array, local_row, local_col)
                    if candidate is not None:
                        values[index] = candidate
                return values

    @staticmethod
    def _sample_points_with_neighborhood(dataset: Any, xs: list[float], ys: list[float]) -> np.ndarray:
        values = np.full(len(xs), np.nan, dtype=np.float32)
        for index, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(dataset.transform, x, y)
            values[index] = LandsatSurfaceTemperatureFusion._nearest_valid_from_dataset(dataset, row, col)
        return values

    @staticmethod
    def _nearest_valid_from_dataset(dataset: Any, row: int, col: int) -> float:
        if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
            return np.nan
        block = dataset.read(1, window=Window(col, row, 1, 1), masked=True)
        candidate = LandsatSurfaceTemperatureFusion._nearest_valid_from_array(block, 0, 0)
        return float(candidate) if candidate is not None else np.nan

    @staticmethod
    def _nearest_valid_from_array(array: Any, center_row: int, center_col: int) -> float | None:
        """Return only the sampled pixel; never borrow a neighboring thermal value."""
        if array.size == 0:
            return None
        mask = np.ma.getmaskarray(array)
        values = np.asarray(array.data, dtype=np.float32)
        values = np.where(mask, np.nan, values)
        height, width = values.shape
        if center_row < 0 or center_row >= height or center_col < 0 or center_col >= width:
            return None
        value = float(values[center_row, center_col])
        if not np.isfinite(value) or value <= 0 or value > 65535:
            return None
        temperature_c = value * ST_SCALE + ST_OFFSET_K - 273.15
        if not ST_MIN_C <= temperature_c <= ST_MAX_C:
            return None
        return value

    def _download(self, href: str) -> bytes:
        request = Request(href, headers={"Accept": "image/tiff, application/octet-stream", "User-Agent": "MANGAI/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError("Landsat ST raster asset is unavailable.", details={"reason": str(exc)}) from exc
