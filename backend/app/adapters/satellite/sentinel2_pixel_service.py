from __future__ import annotations

from typing import Any

from backend.app.adapters.satellite.sentinel2_pixels import PixelExtractionConfig, Sentinel2PixelExtractor
from ml.common.provenance import DataBatch


class Sentinel2PixelService:
    """Coordinate scene discovery output with real pixel extraction."""

    def __init__(self, extractor: Sentinel2PixelExtractor | None = None) -> None:
        self.extractor = extractor or Sentinel2PixelExtractor()

    def ingest_scene(
        self,
        scene: dict[str, Any],
        site_id: str,
        *,
        aoi_bbox: tuple[float, float, float, float] | None = None,
    ) -> DataBatch:
        if not scene.get("scene_id"):
            raise ValueError("Sentinel-2 scene_id is required for pixel ingestion.")
        extractor = self.extractor
        if aoi_bbox is not None:
            extractor = Sentinel2PixelExtractor(
                PixelExtractionConfig(
                    target_resolution_m=extractor.config.target_resolution_m,
                    cloud_mask_keys=extractor.config.cloud_mask_keys,
                    timeout_seconds=extractor.config.timeout_seconds,
                    aoi_bbox=aoi_bbox,
                )
            )
        return extractor.extract_scene(scene, site_id)
