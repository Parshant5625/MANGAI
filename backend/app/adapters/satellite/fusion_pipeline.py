from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from backend.app.adapters.satellite.alignment import parse_scene_datetime
from backend.app.adapters.satellite.landsat_st_fusion import LandsatSurfaceTemperatureFusion
from backend.app.adapters.satellite.planetary_computer import (
    PlanetaryComputerLandsatSurfaceTemperatureProvider,
    PlanetaryComputerSentinel2Provider,
)
from backend.app.adapters.satellite.sentinel2_pixel_service import Sentinel2PixelService
from backend.app.core.errors import DataUnavailableError
from ml.common.external_contracts import validate_external_batch
from ml.common.provenance import DataBatch

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

    def __init__(self, sentinel_provider=None, landsat_provider=None, optical_service=None, thermal_fusion=None) -> None:
        self.sentinel_provider = sentinel_provider or PlanetaryComputerSentinel2Provider()
        self.landsat_provider = landsat_provider or PlanetaryComputerLandsatSurfaceTemperatureProvider(max_items=50)
        self.optical_service = optical_service or Sentinel2PixelService()
        self.thermal_fusion = thermal_fusion or LandsatSurfaceTemperatureFusion()

    def run(self, *, site_id: str, start: str, end: str, latitude: float, longitude: float, max_temporal_days: int = 16, limit: int = 5) -> FusionRunResult:
        if not site_id.strip():
            raise ValueError("site_id must not be empty")
        if start > end:
            raise DataUnavailableError("Satellite fusion start date must not be after end date.")
        if max_temporal_days < 0:
            raise DataUnavailableError("Maximum temporal matching window must be non-negative.")

        sentinel_batch = self.sentinel_provider.search_scenes(site_id, start, end, limit=limit)
        if not sentinel_batch.records:
            raise DataUnavailableError("No Sentinel-2 scenes are available for fusion.")

        start_date = parse_scene_datetime(f"{start}T00:00:00Z").date() - timedelta(days=max_temporal_days)
        end_date = parse_scene_datetime(f"{end}T23:59:59Z").date() + timedelta(days=max_temporal_days)
        thermal_scenes = self.landsat_provider.discover(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
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
        failures: list[dict[str, Any]] = []

        for scene in sentinel_batch.records:
            optical = self.optical_service.ingest_scene(scene, site_id, aoi_bbox=aoi_bbox)
            candidates = self._rank_thermal_candidates(scene.get("datetime"), thermal_scenes, max_days=max_temporal_days)
            if not candidates:
                failures.append({"sentinel_scene_id": scene.get("scene_id"), "error": "no temporal Landsat candidate"})
                continue

            selected_fused = None
            selected_scene = None
            candidate_failures = []
            for candidate in candidates:
                try:
                    fused = self.thermal_fusion.fuse(optical, candidate, aoi_bbox=aoi_bbox)
                    errors = validate_external_batch(fused)
                    if errors:
                        candidate_failures.append({"scene_id": candidate.get("scene_id"), "errors": errors})
                        continue
                    selected_fused = fused
                    selected_scene = candidate
                    break
                except DataUnavailableError as exc:
                    failure: dict[str, Any] = {"scene_id": candidate.get("scene_id"), "error": str(exc)}
                    if exc.details:
                        failure["details"] = exc.details
                    candidate_failures.append(failure)

            if selected_fused is None or selected_scene is None:
                failures.append({"sentinel_scene_id": scene.get("scene_id"), "candidate_failures": candidate_failures})
                continue

            fused_batches.append(selected_fused)
            distances.append(float(selected_scene["temporal_distance_days"]))

        if not fused_batches:
            raise DataUnavailableError(
                "No Sentinel-2 scene could be thermally fused with valid Landsat ST pixels.",
                details={"failures": failures},
            )

        return FusionRunResult(
            self._combine_batches(fused_batches),
            len(sentinel_batch.records),
            len(fused_batches),
            len(thermal_scenes),
            tuple(distances),
        )

    @staticmethod
    def _rank_thermal_candidates(reference_datetime: Any, scenes: list[dict[str, Any]], *, max_days: int) -> list[dict[str, Any]]:
        reference = parse_scene_datetime(reference_datetime)
        candidates = []
        for scene in scenes:
            try:
                scene_time = parse_scene_datetime(scene.get("datetime"))
            except DataUnavailableError:
                continue
            distance_days = abs((scene_time - reference).total_seconds()) / 86400.0
            if distance_days <= max_days:
                selected = dict(scene)
                selected["temporal_distance_days"] = distance_days
                candidates.append((distance_days, selected))
        candidates.sort(key=lambda item: (item[0], str(item[1].get("scene_id", ""))))
        return [scene for _, scene in candidates]

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
