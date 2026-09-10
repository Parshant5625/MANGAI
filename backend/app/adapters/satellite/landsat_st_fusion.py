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
except ImportError:  # pragma: no cover
    rasterio = None


ST_SCALE = 0.00341802
ST_OFFSET_K = 149.0


class LandsatSurfaceTemperatureFusion:
    """Sample real Landsat Collection 2 ST at Sentinel-2 pixel locations."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if rasterio is None:
            raise DataUnavailableError("Landsat ST fusion requires rasterio to be installed.")
        if timeout_seconds <= 0:
            raise DataUnavailableError("Landsat ST timeout must be positive.")
        self.timeout_seconds = timeout_seconds

    def fuse(self, optical_batch: DataBatch, thermal_scene: dict[str, Any]) -> DataBatch:
        if optical_batch.provenance.mode != "live":
            raise DataUnavailableError("Thermal fusion is only enabled for live satellite batches.")
        if optical_batch.provenance.dataset != "satellite_features":
            raise DataUnavailableError("Thermal fusion expects satellite_features records.")

        asset = ((thermal_scene.get("assets") or {}).get("surface_temperature"))
        if not asset:
            raise DataUnavailableError("Landsat ST scene has no surface temperature asset.")

        raw = self._download(str(asset))
        try:
            with MemoryFile(BytesIO(raw)) as memory_file:
                with memory_file.open() as dataset:
                    if dataset.crs is None:
                        raise DataUnavailableError("Landsat ST raster is missing CRS information.")
                    if not optical_batch.records:
                        raise DataUnavailableError("Cannot fuse thermal data into an empty optical batch.")

                    source_crs = optical_batch.records[0].get("crs")
                    if not source_crs:
                        raise DataUnavailableError("Sentinel-2 records are missing CRS information.")
                    xs = [float(record["x"]) for record in optical_batch.records]
                    ys = [float(record["y"]) for record in optical_batch.records]
                    target_x, target_y = transform(source_crs, dataset.crs, xs, ys)
                    samples = list(dataset.sample(zip(target_x, target_y), indexes=1))
                    thermal_values = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        except DataUnavailableError:
            raise
        except Exception as exc:  # rasterio driver/IO errors vary by environment
            raise DataUnavailableError(
                "Landsat ST raster could not be decoded or sampled.",
                details={"reason": str(exc)},
            ) from exc

        valid = np.isfinite(thermal_values) & (thermal_values > 0) & (thermal_values <= 65535)
        if not valid.any():
            raise DataUnavailableError("Landsat ST scene has no valid thermal pixels at Sentinel locations.")

        records: list[dict[str, Any]] = []
        for record, dn, is_valid in zip(optical_batch.records, thermal_values, valid):
            if not is_valid:
                continue
            updated = dict(record)
            updated["land_surface_temperature"] = float(dn * ST_SCALE + ST_OFFSET_K - 273.15)
            updated["thermal_source"] = "landsat-collection-2-surface-temperature"
            updated["thermal_scene_id"] = thermal_scene.get("scene_id")
            records.append(updated)

        coverage = len(records) / len(optical_batch.records)
        ingested_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(
            json.dumps(records, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        provenance = DataProvenance(
            source_name="Copernicus Sentinel-2 L2A + USGS Landsat Collection 2 ST",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=thermal_scene.get("datetime") or optical_batch.provenance.acquired_at,
            ingested_at=ingested_at,
            source_version=f"{optical_batch.provenance.source_version or 'sentinel-2'}+{thermal_scene.get('collection', 'landsat-c2l2-st')}",
            source_uri=str(thermal_scene.get("source_uri") or thermal_scene.get("scene_id") or "usgs-landsat-st"),
            checksum=checksum,
            license_note="Sentinel-2 and Landsat source terms must be verified for deployment; USGS Landsat Collection 2 is publicly accessible.",
            quality_score=coverage,
            row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)

    def _download(self, href: str) -> bytes:
        request = Request(
            href,
            headers={"Accept": "image/tiff, application/octet-stream", "User-Agent": "MANGAI/1.0"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DataUnavailableError(
                "Landsat ST raster asset is unavailable.",
                details={"reason": str(exc)},
            ) from exc
