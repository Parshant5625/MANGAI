from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.app.adapters.satellite.alignment import select_temporally_matched_scene
from backend.app.adapters.satellite.landsat_st_fusion import LandsatSurfaceTemperatureFusion
from backend.app.adapters.satellite.planetary_computer import (
    PlanetaryComputerLandsatSurfaceTemperatureProvider,
    PlanetaryComputerSentinel2Provider,
)
from backend.app.adapters.satellite.sentinel2_pixel_service import Sentinel2PixelService
from backend.app.core.errors import DataUnavailableError
from ml.common.external_contracts import validate_external_batch
from ml.common.provenance import DataBatch


# Small bounded AOI around the requested mine/site coordinate.  This keeps
# Planetary Computer COG reads fast while covering the normal 500 m geology
# matching radius with margin.
LIVE_AOI_HALF_DEG = 0.02


class SentinelSceneProvider(Protocol):
    def search_scenes(self, site_id: str, start: str, end: str, limit: int = 10) -> DataBatch: ...


class LandsatSceneProvider(Protocol):
    def discover(
        self,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class FusionRunResult:
    batch: DataBatch
    sentinel_scene_count: int
    fused_scene_count: int
    thermal_scene_count: int
    temporal_distance_days: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch.to_dict(),
            "sentinel_scene_count": self.sentinel_scene_count,
            "fused_scene_count": self.fused_scene_count,
            "thermal_scene_count": self.thermal_scene_count,
            "temporal_distance_days": list(self.temporal_distance_days),
        }


class LiveSatelliteFusionPipeline:
    """Fail-closed Sentinel-2 -> Landsat ST fusion using Planetary Computer."""

    def __init__(
        self,
        sentinel_provider: SentinelSceneProvider | None = None,
        landsat_provider: LandsatSceneProvider | None = None,
        optical_service: Sentinel2PixelService | None = None,
        thermal_fusion: LandsatSurfaceTemperatureFusion | None = None,
    ) -> None:
        self.sentinel_provider = sentinel_provider or PlanetaryComputerSentinel2Provider()
        self.landsat_provider = landsat_provider or PlanetaryComputerLandsatSurfaceTemperatureProvider()
        self.optical_service = optical_service or Sentinel2PixelService()
        self.thermal_fusion = thermal_fusion or LandsatSurfaceTemperatureFusion()

    def run(
        self,
        *,
        site_id: str,
        start: str,
        end: str,
        latitude: float,
        longitude: float,
        max_temporal_days: int = 16,
        limit: int = 5,
    ) -> FusionRunResult:
        if not site_id.strip():
            raise ValueError("site_id must not be empty")
        if start > end:
            raise DataUnavailableError("Satellite fusion start date must not be after end date.")
        if max_temporal_days < 0:
            raise DataUnavailableError("Maximum temporal matching window must be non-negative.")

        sentinel_batch = self.sentinel_provider.search_scenes(site_id, start, end, limit=limit)
        if not sentinel_batch.records:
            raise DataUnavailableError("No Sentinel-2 scenes are available for fusion.")

        thermal_scenes = self.landsat_provider.discover(
            latitude=latitude,
            longitude=longitude,
            start_date=start,
            end_date=end,
        )
        if not thermal_scenes:
            raise DataUnavailableError("No Landsat ST scenes are available for fusion.")

        half_deg = LIVE_AOI_HALF_DEG
        aoi_bbox = (
            max(-180.0, longitude - half_deg),
            max(-90.0, latitude - half_deg),
            min(180.0, longitude + half_deg),
            min(90.0, latitude + half_deg),
        )

        fused_batches: list[DataBatch] = []
        distances: list[float] = []
        for scene in sentinel_batch.records:
            optical = self.optical_service.ingest_scene(scene, site_id, aoi_bbox=aoi_bbox)
            matched = select_temporally_matched_scene(
                scene.get("datetime"),
                thermal_scenes,
                max_days=max_temporal_days,
            )
            fused = self.thermal_fusion.fuse(optical, matched)
            errors = validate_external_batch(fused)
            if errors:
                raise DataUnavailableError(
                    "Fused satellite batch failed the canonical data contract.",
                    details={"errors": errors, "scene_id": scene.get("scene_id")},
                )
            fused_batches.append(fused)
            distances.append(float(matched["temporal_distance_days"]))

        combined = self._combine_batches(fused_batches)
        return FusionRunResult(
            combined,
            len(sentinel_batch.records),
            len(fused_batches),
            len(thermal_scenes),
            tuple(distances),
        )

    @staticmethod
    def _combine_batches(batches: list[DataBatch]) -> DataBatch:
        if not batches:
            raise DataUnavailableError("Cannot combine an empty satellite fusion result.")
        records = [record for batch in batches for record in batch.records]
        first = batches[0].provenance
        from ml.common.provenance import DataProvenance

        provenance = DataProvenance(
            source_name=first.source_name,
            source_kind=first.source_kind,
            mode=first.mode,
            dataset=first.dataset,
            acquired_at=max(batch.provenance.acquired_at or "" for batch in batches),
            ingested_at=max(batch.provenance.ingested_at or "" for batch in batches),
            source_version=first.source_version,
            source_uri=first.source_uri,
            checksum=first.checksum,
            license_note=first.license_note,
            quality_score=sum(batch.provenance.quality_score for batch in batches) / len(batches),
            row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)
