from __future__ import annotations

from typing import Any

from backend.app.adapters.satellite.sentinel2_pixels import Sentinel2PixelExtractor
from ml.common.provenance import DataBatch


class Sentinel2PixelService:
    """Coordinate scene discovery output with real pixel extraction."""

    def __init__(self, extractor: Sentinel2PixelExtractor | None = None) -> None:
        self.extractor = extractor or Sentinel2PixelExtractor()

    def ingest_scene(self, scene: dict[str, Any], site_id: str) -> DataBatch:
        if not scene.get("scene_id"):
            raise ValueError("Sentinel-2 scene_id is required for pixel ingestion.")
        return self.extractor.extract_scene(scene, site_id)
