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
ST_QA_SCALE_K = 0.01
MAX_ST_UNCERTAINTY_K = 2.0
QA_PIXEL_FILL = 1 << 0
QA_PIXEL_DILATED_CLOUD = 1 << 1
QA_PIXEL_CIRRUS = 1 << 2
QA_PIXEL_CLOUD = 1 << 3
QA_PIXEL_CLOUD_SHADOW = 1 << 4
QA_PIXEL_SNOW = 1 << 5
QA_RADSAT_DROPPED_PIXEL = 1 << 9
QA_RADSAT_TERRAIN_OCCLUSION = 1 << 11


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
        assets = thermal_scene.get("assets") or {}
        asset = assets.get("surface_temperature")
        if not asset:
            raise DataUnavailableError("Landsat ST scene has no surface temperature asset.")
        if not optical_batch.records:
            raise DataUnavailableError("Cannot fuse thermal data into an empty optical batch.")

        href = str(asset)
        try:
            if aoi_bbox is not None and href.startswith(("http://", "https://")):
                thermal_values, uncertainty_k, qa_rejected = self._sample_remote_cog(href, optical_batch, aoi_bbox, assets)
            else:
                raw = self._download(href)
                with MemoryFile(BytesIO(raw)) as memory_file:
                    with memory_file.open() as dataset:
                        thermal_values = self._sample_dataset(dataset, optical_batch)
                uncertainty_k = np.full(len(thermal_values), np.nan, dtype=np.float32)
                qa_rejected = np.zeros(len(thermal_values), dtype=bool)
        except DataUnavailableError:
            raise
        except Exception as exc:
            raise DataUnavailableError(
                "Landsat ST raster could not be decoded or sampled.",
                details={"reason": f"{type(exc).__name__}: {exc}", "asset": href.split("?")[0]},
            ) from exc

        temperatures_c = thermal_values * ST_SCALE + ST_OFFSET_K - 273.15
        valid = (
            np.isfinite(thermal_values)
            & (thermal_values > 0)
            & (thermal_values <= 65535)
            & np.isfinite(temperatures_c)
            & (temperatures_c >= ST_MIN_C)
            & (temperatures_c <= ST_MAX_C)
            & ~qa_rejected
        )
        if not valid.any():
            raise DataUnavailableError(
                "Landsat ST scene has no valid thermal pixels at Sentinel locations.",
                details={"temperature_range_c": [ST_MIN_C, ST_MAX_C], "qa_rejected": int(qa_rejected.sum())},
            )

        records: list[dict[str, Any]] = []
        for record, temperature_c, is_valid, uncertainty in zip(optical_batch.records, temperatures_c, valid, uncertainty_k):
            if not is_valid:
                continue
            updated = dict(record)
            updated["land_surface_temperature"] = float(temperature_c)
            updated["thermal_source"] = "landsat-collection-2-surface-temperature"
            updated["thermal_scene_id"] = thermal_scene.get("scene_id")
            updated["thermal_resampling"] = "nearest-pixel"
            if np.isfinite(uncertainty):
                updated["thermal_uncertainty_k"] = float(uncertainty)
            records.append(updated)

        coverage = len(records) / len(optical_batch.records)
        quality = coverage
        if records and any("thermal_uncertainty_k" in record for record in records):
            uncertainty_values = np.array([record["thermal_uncertainty_k"] for record in records], dtype=np.float32)
            quality *= float(np.clip(1.0 - uncertainty_values / MAX_ST_UNCERTAINTY_K, 0.0, 1.0).mean())
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        provenance = DataProvenance(
            source_name="Copernicus Sentinel-2 L2A + USGS Landsat Collection 2 ST via Microsoft Planetary Computer",
            source_kind="satellite", mode="live", dataset="satellite_features",
            acquired_at=thermal_scene.get("datetime") or optical_batch.provenance.acquired_at,
            ingested_at=datetime.now(UTC).isoformat(),
            source_version=f"{optical_batch.provenance.source_version or 'sentinel-2'}+{thermal_scene.get('collection', 'landsat-c2-l2')}",
            source_uri=str(thermal_scene.get("source_uri") or thermal_scene.get("scene_id") or "microsoft-planetary-computer"),
            checksum=checksum,
            license_note="Sentinel-2 and Landsat source terms must be verified for deployment; USGS Landsat Collection 2 is publicly accessible.",
            quality_score=max(0.0, min(1.0, quality)), row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)

    def _sample_dataset(self, dataset: Any, optical_batch: DataBatch) -> np.ndarray:
        if dataset.crs is None:
            raise DataUnavailableError("Landsat ST raster is missing CRS information.")
        source_crs = optical_batch.records[0].get("crs")
        if not source_crs:
            raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
        xs = [float(r["x"]) for r in optical_batch.records]
        ys = [float(r["y"]) for r in optical_batch.records]
        target_x, target_y = transform(source_crs, dataset.crs, xs, ys)
        return self._sample_points(dataset, target_x, target_y)

    def _sample_remote_cog(self, href: str, optical_batch: DataBatch, aoi_bbox: tuple[float, float, float, float], assets: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        env_options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
            "GDAL_HTTP_VERSION": "1.1", "GDAL_HTTP_MULTIPLEX": "NO", "GDAL_HTTP_MAX_RETRY": "4",
            "GDAL_HTTP_RETRY_DELAY": "1", "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "5000000",
        }
        qa_pixel_href = assets.get("qa_pixel")
        st_qa_href = assets.get("st_qa")
        qa_radsat_href = assets.get("qa_radsat")
        if not qa_pixel_href or not st_qa_href:
            raise DataUnavailableError("Landsat ST scene is missing required QA_PIXEL or ST_QA assets.", details={"required_assets": ["qa_pixel", "st_qa"]})

        with rasterio.Env(**env_options):
            with rasterio.open(href, sharing=False) as dataset:
                if dataset.crs is None:
                    raise DataUnavailableError("Landsat ST raster is missing CRS information.")
                source_crs = optical_batch.records[0].get("crs")
                if not source_crs:
                    raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
                left, bottom, right, top = transform_bounds("EPSG:4326", dataset.crs, *aoi_bbox, densify_pts=21)
                window = from_bounds(left, bottom, right, top, transform=dataset.transform).intersection(Window(0, 0, dataset.width, dataset.height))
                if window.width <= 0 or window.height <= 0:
                    raise DataUnavailableError("Landsat ST raster does not overlap the requested AOI.")
                array = dataset.read(1, window=window, masked=True)
                xs = [float(r["x"]) for r in optical_batch.records]
                ys = [float(r["y"]) for r in optical_batch.records]
                target_x, target_y = transform(source_crs, dataset.crs, xs, ys)
                values = self._sample_window(array, dataset.transform, window, target_x, target_y)

            qa_pixel, qa_pixel_transform, qa_pixel_crs = self._read_remote_window(qa_pixel_href, aoi_bbox, env_options)
            st_qa, st_qa_transform, st_qa_crs = self._read_remote_window(st_qa_href, aoi_bbox, env_options)
            qa_radsat = qa_radsat_transform = qa_radsat_crs = None
            if qa_radsat_href:
                qa_radsat, qa_radsat_transform, qa_radsat_crs = self._read_remote_window(qa_radsat_href, aoi_bbox, env_options)

            qa_rejected = np.zeros(len(values), dtype=bool)
            uncertainty_k = np.full(len(values), np.nan, dtype=np.float32)
            qa_x, qa_y = transform(source_crs, qa_pixel_crs, xs, ys)
            stqa_x, stqa_y = transform(source_crs, st_qa_crs, xs, ys)
            qa_stats = {"pixel_quality_rejected": 0, "st_uncertainty_rejected": 0, "radsat_rejected": 0, "qa_unreadable": 0}
            for index, (qx, qy, ux, uy) in enumerate(zip(qa_x, qa_y, stqa_x, stqa_y)):
                qrow, qcol = rasterio.transform.rowcol(qa_pixel_transform, qx, qy)
                qvalue = self._value_from_array(qa_pixel, int(qrow), int(qcol))
                if qvalue is None:
                    qa_rejected[index] = True
                    qa_stats["qa_unreadable"] += 1
                    continue
                if self._qa_pixel_is_bad(int(qvalue)):
                    qa_rejected[index] = True
                    qa_stats["pixel_quality_rejected"] += 1
                    continue
                srow, scol = rasterio.transform.rowcol(st_qa_transform, ux, uy)
                uncertainty_dn = self._value_from_array(st_qa, int(srow), int(scol))
                if uncertainty_dn is None:
                    qa_rejected[index] = True
                    qa_stats["qa_unreadable"] += 1
                    continue
                uncertainty = float(uncertainty_dn) * ST_QA_SCALE_K
                if not np.isfinite(uncertainty) or uncertainty > MAX_ST_UNCERTAINTY_K:
                    qa_rejected[index] = True
                    qa_stats["st_uncertainty_rejected"] += 1
                    continue
                uncertainty_k[index] = uncertainty
                if qa_radsat is not None and qa_radsat_crs is not None:
                    rxs, rys = transform(source_crs, qa_radsat_crs, [xs[index]], [ys[index]])
                    rrow, rcol = rasterio.transform.rowcol(qa_radsat_transform, rxs[0], rys[0])
                    rvalue = self._value_from_array(qa_radsat, int(rrow), int(rcol))
                    if rvalue is None or self._qa_radsat_is_bad(int(rvalue)):
                        qa_rejected[index] = True
                        qa_stats["radsat_rejected"] += 1
            self._last_qa_stats = qa_stats
            return values, uncertainty_k, qa_rejected

    @staticmethod
    def _read_remote_window(href: str, aoi_bbox: tuple[float, float, float, float], env_options: dict[str, str]) -> tuple[Any, Any, Any]:
        with rasterio.Env(**env_options):
            with rasterio.open(str(href), sharing=False) as dataset:
                if dataset.crs is None:
                    raise DataUnavailableError("Landsat QA raster is missing CRS information.")
                left, bottom, right, top = transform_bounds("EPSG:4326", dataset.crs, *aoi_bbox, densify_pts=21)
                window = from_bounds(left, bottom, right, top, transform=dataset.transform).intersection(Window(0, 0, dataset.width, dataset.height))
                if window.width <= 0 or window.height <= 0:
                    raise DataUnavailableError("Landsat QA raster does not overlap the requested AOI.")
                local_transform = dataset.transform * rasterio.Affine.translation(window.col_off, window.row_off)
                return dataset.read(1, window=window, masked=True), local_transform, dataset.crs

    @staticmethod
    def _sample_points(dataset: Any, xs: list[float], ys: list[float]) -> np.ndarray:
        values = np.full(len(xs), np.nan, dtype=np.float32)
        for index, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(dataset.transform, x, y)
            value = LandsatSurfaceTemperatureFusion._value_from_dataset(dataset, row, col)
            if value is not None:
                values[index] = value
        return values

    @staticmethod
    def _sample_window(array: Any, transform_: Any, window: Any, xs: list[float], ys: list[float]) -> np.ndarray:
        values = np.full(len(xs), np.nan, dtype=np.float32)
        for index, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(transform_, x, y)
            value = LandsatSurfaceTemperatureFusion._value_from_array(array, int(row) - int(window.row_off), int(col) - int(window.col_off))
            if value is not None:
                values[index] = value
        return values

    @staticmethod
    def _value_from_dataset(dataset: Any, row: int, col: int) -> float | None:
        if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
            return None
        return LandsatSurfaceTemperatureFusion._value_from_array(dataset.read(1, window=Window(col, row, 1, 1), masked=True), 0, 0)

    @staticmethod
    def _value_from_array(array: Any, row: int, col: int) -> float | None:
        if array.size == 0 or row < 0 or row >= array.shape[0] or col < 0 or col >= array.shape[1]:
            return None
        if np.ma.getmaskarray(array)[row, col]:
            return None
        value = float(np.asarray(array.data, dtype=np.float32)[row, col])
        return value if np.isfinite(value) else None

    @staticmethod
    def _qa_pixel_is_bad(value: int) -> bool:
        return (value & (QA_PIXEL_FILL | QA_PIXEL_DILATED_CLOUD | QA_PIXEL_CIRRUS | QA_PIXEL_CLOUD | QA_PIXEL_CLOUD_SHADOW | QA_PIXEL_SNOW)) != 0

    @staticmethod
    def _qa_radsat_is_bad(value: int) -> bool:
        return (value & (QA_RADSAT_DROPPED_PIXEL | QA_RADSAT_TERRAIN_OCCLUSION)) != 0

    @staticmethod
    def _nearest_valid_from_array(array: Any, center_row: int, center_col: int) -> float | None:
        return LandsatSurfaceTemperatureFusion._value_from_array(array, center_row, center_col)

    def _download(self, href: str) -> bytes:
        request = Request(href, headers={"Accept": "image/tiff, application/octet-stream", "User-Agent": "MANGAI/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError("Landsat ST raster asset is unavailable.", details={"reason": str(exc)}) from exc
