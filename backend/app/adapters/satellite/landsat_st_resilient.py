from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np

from backend.app.adapters.satellite.landsat_st_fusion import (
    ST_MAX_C,
    ST_MIN_C,
    ST_OFFSET_K,
    ST_SCALE,
    LandsatSurfaceTemperatureFusion,
)
from backend.app.core.errors import DataUnavailableError
from ml.common.provenance import DataBatch, DataProvenance

try:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform, transform_bounds
    from rasterio.windows import Window, from_bounds
except ImportError:  # pragma: no cover
    rasterio = None


class ResilientLandsatSurfaceTemperatureFusion(LandsatSurfaceTemperatureFusion):
    """Strict Landsat ST fusion with an explicitly demo-only degraded fallback.

    Production remains fail-closed. The fallback is enabled only when
    MANGAI_ALLOW_DEGRADED_THERMAL=1 and uses the physical ST validity range,
    while clearly marking the resulting provenance as QA-degraded.
    """

    def __init__(self, timeout_seconds: float = 60.0, allow_degraded: bool | None = None) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.allow_degraded = (
            os.getenv("MANGAI_ALLOW_DEGRADED_THERMAL") == "1"
            if allow_degraded is None
            else allow_degraded
        )

    def fuse(
        self,
        optical_batch: DataBatch,
        thermal_scene: dict[str, Any],
        *,
        aoi_bbox: tuple[float, float, float, float] | None = None,
    ) -> DataBatch:
        try:
            return super().fuse(optical_batch, thermal_scene, aoi_bbox=aoi_bbox)
        except DataUnavailableError as strict_error:
            if not self.allow_degraded or aoi_bbox is None:
                raise
            return self._degraded_fuse(
                optical_batch,
                thermal_scene,
                aoi_bbox=aoi_bbox,
                strict_error=str(strict_error),
            )

    def _degraded_fuse(
        self,
        optical_batch: DataBatch,
        thermal_scene: dict[str, Any],
        *,
        aoi_bbox: tuple[float, float, float, float],
        strict_error: str,
    ) -> DataBatch:
        if rasterio is None:
            raise DataUnavailableError("Degraded Landsat ST fusion requires rasterio.")
        href = str((thermal_scene.get("assets") or {}).get("surface_temperature") or "")
        if not href.startswith(("http://", "https://")):
            raise DataUnavailableError("Degraded Landsat ST fallback requires a remote COG asset.")

        env_options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
            "GDAL_HTTP_VERSION": "1.1",
            "GDAL_HTTP_MULTIPLEX": "NO",
            "GDAL_HTTP_MAX_RETRY": "4",
            "GDAL_HTTP_RETRY_DELAY": "1",
            "VSI_CACHE": "TRUE",
            "VSI_CACHE_SIZE": "5000000",
        }
        records: list[dict[str, Any]] = []
        with rasterio.Env(**env_options):
            with rasterio.open(href, sharing=False) as dataset:
                if dataset.crs is None:
                    raise DataUnavailableError("Landsat ST raster is missing CRS information.")
                source_crs = optical_batch.records[0].get("crs")
                if not source_crs:
                    raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", dataset.crs, *aoi_bbox, densify_pts=21
                )
                window = from_bounds(left, bottom, right, top, transform=dataset.transform).intersection(
                    Window(0, 0, dataset.width, dataset.height)
                )
                if window.width <= 0 or window.height <= 0:
                    raise DataUnavailableError("Landsat ST raster does not overlap the requested AOI.")
                array = dataset.read(1, window=window, masked=True)
                xs = [float(record["x"]) for record in optical_batch.records]
                ys = [float(record["y"]) for record in optical_batch.records]
                target_x, target_y = transform(source_crs, dataset.crs, xs, ys)
                for record, x, y in zip(optical_batch.records, target_x, target_y):
                    absolute_row, absolute_col = rowcol(dataset.transform, x, y)
                    local_row = int(absolute_row) - int(window.row_off)
                    local_col = int(absolute_col) - int(window.col_off)
                    if local_row < 0 or local_col < 0 or local_row >= array.shape[0] or local_col >= array.shape[1]:
                        continue
                    if np.ma.getmaskarray(array)[local_row, local_col]:
                        continue
                    dn = float(np.asarray(array.data, dtype=np.float32)[local_row, local_col])
                    temperature_c = dn * ST_SCALE + ST_OFFSET_K - 273.15
                    if not np.isfinite(temperature_c) or not (ST_MIN_C <= temperature_c <= ST_MAX_C):
                        continue
                    updated = dict(record)
                    updated["land_surface_temperature"] = float(temperature_c)
                    updated["thermal_source"] = "landsat-collection-2-surface-temperature"
                    updated["thermal_scene_id"] = thermal_scene.get("scene_id")
                    updated["thermal_resampling"] = "nearest-pixel"
                    updated["thermal_quality_mode"] = "qa-degraded-demo"
                    updated["thermal_qa_note"] = "Strict QA rejected all sampled pixels; demo fallback used physical ST validity only."
                    records.append(updated)

        if not records:
            raise DataUnavailableError(
                "Landsat ST scene has no physically valid thermal pixels at Sentinel locations.",
                details={"strict_error": strict_error, "temperature_range_c": [ST_MIN_C, ST_MAX_C]},
            )

        checksum = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        provenance = DataProvenance(
            source_name="USGS Landsat Collection 2 ST via Microsoft Planetary Computer (demo QA-degraded fallback)",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=thermal_scene.get("datetime") or optical_batch.provenance.acquired_at,
            ingested_at=datetime.now(UTC).isoformat(),
            source_version=str(thermal_scene.get("collection", "landsat-c2-l2")),
            source_uri=str(thermal_scene.get("source_uri") or thermal_scene.get("scene_id") or href),
            checksum=checksum,
            license_note="DEMO ONLY: strict Landsat QA was unavailable/rejected at sampled locations; replace with QA-valid fusion for production inference.",
            quality_score=max(0.0, min(1.0, len(records) / len(optical_batch.records) * 0.5)),
            row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)
