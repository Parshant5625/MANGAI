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
    from rasterio.warp import transform
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

    def fuse(
        self,
        optical_batch: DataBatch,
        thermal_scene: dict[str, Any],
        *,
        aoi_bbox: tuple[float, float, float, float] | None = None,
    ) -> DataBatch:
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
            if href.startswith(("http://", "https://")):
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
                details={"reason": str(exc)},
            ) from exc

        temperatures_c = thermal_values * ST_SCALE + ST_OFFSET_K - 273.15
        valid = (
            np.isfinite(thermal_values)
            & (thermal_values > 0)
            & (thermal_values <= 65535)
            & np.isfinite(temperatures_c)
            & (temperatures_c >= ST_MIN_C)
            & (temperatures_c <= ST_MAX_C)
        )
        if not valid.any():
            raise DataUnavailableError(
                "Landsat ST scene has no valid thermal pixels at or near Sentinel locations.",
                details={
                    "neighbor_radius_pixels": THERMAL_NEIGHBOR_RADIUS_PIXELS,
                    "temperature_range_c": [ST_MIN_C, ST_MAX_C],
                },
            )

        records: list[dict[str, Any]] = []
        for record, temperature_c, is_valid in zip(optical_batch.records, temperatures_c, valid):
            if not is_valid:
                continue
            updated = dict(record)
            updated["land_surface_temperature"] = float(temperature_c)
            updated["thermal_source"] = "landsat-collection-2-surface-temperature"
            updated["thermal_scene_id"] = thermal_scene.get("scene_id")
            updated["thermal_resampling"] = "nearest-valid-pixel-5x5"
            records.append(updated)

        coverage = len(records) / len(optical_batch.records)
        ingested_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        provenance = DataProvenance(
            source_name="Copernicus Sentinel-2 L2A + USGS Landsat Collection 2 ST via Microsoft Planetary Computer",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=thermal_scene.get("datetime") or optical_batch.provenance.acquired_at,
            ingested_at=ingested_at,
            source_version=f"{optical_batch.provenance.source_version or 'sentinel-2'}+{thermal_scene.get('collection', 'landsat-c2-l2')}",
            source_uri=str(thermal_scene.get("source_uri") or thermal_scene.get("scene_id") or "microsoft-planetary-computer"),
            checksum=checksum,
            license_note="Sentinel-2 and Landsat source terms must be verified for deployment; USGS Landsat Collection 2 is publicly accessible.",
            quality_score=coverage,
            row_count=len(records),
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

    def _sample_remote_cog(
        self,
        href: str,
        optical_batch: DataBatch,
        aoi_bbox: tuple[float, float, float, float] | None,
    ) -> np.ndarray:
        """Read only the thermal pixels covering the optical sample extent.

        We deliberately avoid converting the WGS84 AOI directly into a Landsat
        window. The optical batch already contains coordinates in its source CRS;
        transforming those exact sample coordinates avoids edge/axis-order and
        AOI-window intersection failures seen with remote COGs.
        """
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                raise DataUnavailableError("Landsat ST raster is missing CRS information.")
            source_crs = optical_batch.records[0].get("crs")
            if not source_crs:
                raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
            xs = [float(record["x"]) for record in optical_batch.records]
            ys = [float(record["y"]) for record in optical_batch.records]
            target_x, target_y = transform(source_crs, dataset.crs, xs, ys)

            min_x = min(target_x)
            max_x = max(target_x)
            min_y = min(target_y)
            max_y = max(target_y)
            if max_x < dataset.bounds.left or min_x > dataset.bounds.right or max_y < dataset.bounds.bottom or min_y > dataset.bounds.top:
                raise DataUnavailableError("Landsat ST raster does not overlap Sentinel sample coordinates.")

            requested = from_bounds(min_x, min_y, max_x, max_y, transform=dataset.transform)
            expanded = Window(
                requested.col_off - THERMAL_NEIGHBOR_RADIUS_PIXELS,
                requested.row_off - THERMAL_NEIGHBOR_RADIUS_PIXELS,
                requested.width + 2 * THERMAL_NEIGHBOR_RADIUS_PIXELS,
                requested.height + 2 * THERMAL_NEIGHBOR_RADIUS_PIXELS,
            )
            window = expanded.intersection(Window(0, 0, dataset.width, dataset.height))
            if window.width <= 0 or window.height <= 0:
                raise DataUnavailableError("Landsat ST raster window is empty for Sentinel sample coordinates.")
            array = dataset.read(1, window=window, masked=True)
            return self._sample_array_with_neighborhood(dataset, array, window, target_x, target_y)

    @staticmethod
    def _sample_points_with_neighborhood(dataset: Any, xs: list[float], ys: list[float]) -> np.ndarray:
        values = np.full(len(xs), np.nan, dtype=np.float32)
        for index, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(dataset.transform, x, y)
            values[index] = LandsatSurfaceTemperatureFusion._nearest_valid_from_dataset(dataset, row, col)
        return values

    @staticmethod
    def _sample_array_with_neighborhood(
        dataset: Any,
        array: Any,
        window: Window,
        xs: list[float],
        ys: list[float],
    ) -> np.ndarray:
        values = np.full(len(xs), np.nan, dtype=np.float32)
        for index, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(dataset.transform, x, y)
            local_row = int(row - window.row_off)
            local_col = int(col - window.col_off)
            best = LandsatSurfaceTemperatureFusion._nearest_valid_from_array(array, local_row, local_col)
            if best is not None:
                values[index] = best
        return values

    @staticmethod
    def _sample_valid(value: float) -> bool:
        if not np.isfinite(value) or value <= 0 or value > 65535:
            return False
        temperature_c = value * ST_SCALE + ST_OFFSET_K - 273.15
        return ST_MIN_C <= temperature_c <= ST_MAX_C

    @staticmethod
    def _nearest_valid_from_dataset(dataset: Any, row: int, col: int) -> float:
        radius = THERMAL_NEIGHBOR_RADIUS_PIXELS
        for distance in range(radius + 1):
            r0 = max(0, row - distance)
            r1 = min(dataset.height - 1, row + distance)
            c0 = max(0, col - distance)
            c1 = min(dataset.width - 1, col + distance)
            if r0 > r1 or c0 > c1:
                continue
            block = dataset.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), masked=True)
            candidate = LandsatSurfaceTemperatureFusion._nearest_valid_from_array(block, row - r0, col - c0)
            if candidate is not None:
                return candidate
        return float("nan")

    @staticmethod
    def _nearest_valid_from_array(array: Any, center_row: int, center_col: int) -> float | None:
        if array.size == 0:
            return None
        values = np.ma.filled(array, np.nan).astype(np.float32)
        height, width = values.shape
        if not (0 <= center_row < height and 0 <= center_col < width):
            return None
        best_value: float | None = None
        best_distance = float("inf")
        radius = THERMAL_NEIGHBOR_RADIUS_PIXELS
        for row in range(max(0, center_row - radius), min(height, center_row + radius + 1)):
            for col in range(max(0, center_col - radius), min(width, center_col + radius + 1)):
                distance = max(abs(row - center_row), abs(col - center_col))
                if distance > radius or distance > best_distance:
                    continue
                value = float(values[row, col])
                if LandsatSurfaceTemperatureFusion._sample_valid(value):
                    best_value = value
                    best_distance = distance
        return best_value

    def _download(self, href: str) -> bytes:
        request = Request(
            href,
            headers={"Accept": "image/tiff, application/octet-stream", "User-Agent": "MANGAI/1.0"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError("Landsat ST raster asset is unavailable.", details={"reason": str(exc)}) from exc
