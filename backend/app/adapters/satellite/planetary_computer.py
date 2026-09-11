from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError
from ml.common.provenance import DataBatch, DataProvenance

try:
    import planetary_computer
    import pystac_client
except ImportError:  # pragma: no cover
    planetary_computer = None
    pystac_client = None


PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"
LANDSAT_COLLECTION = "landsat-c2-l2"
SENTINEL2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


class _PlanetaryComputerBase:
    def __init__(self, stac_url: str = PLANETARY_COMPUTER_STAC_URL) -> None:
        if pystac_client is None or planetary_computer is None:
            raise DataUnavailableError("Microsoft Planetary Computer access requires pystac-client and planetary-computer.", details={"dependencies": ["pystac-client", "planetary-computer"]})
        self.stac_url = stac_url.rstrip("/")
        try:
            self.catalog = pystac_client.Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)
        except Exception as exc:
            raise DataUnavailableError("Microsoft Planetary Computer STAC is unavailable.", details={"provider": "microsoft-planetary-computer", "reason": str(exc)}) from exc

    @staticmethod
    def _signed_assets(item: Any) -> dict[str, str]:
        return {str(key): str(asset.href) for key, asset in item.assets.items() if asset.href}


class PlanetaryComputerSentinel2Provider(_PlanetaryComputerBase):
    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self.collection = settings.sentinel2_collection or SENTINEL2_COLLECTION
        self.max_cloud_cover = settings.sentinel2_max_cloud_cover
        self.lat = settings.sentinel2_latitude
        self.lon = settings.sentinel2_longitude
        self.bbox_delta = settings.sentinel2_bbox_delta

    def search_scenes(self, site_id: str, start: str, end: str, limit: int = 10) -> DataBatch:
        if self.lat is None or self.lon is None:
            raise DataUnavailableError("Live Sentinel-2 requires coordinates.", details={"provider": "microsoft-planetary-computer", "configuration": "missing_coordinates"})
        bbox = [self.lon - self.bbox_delta, self.lat - self.bbox_delta, self.lon + self.bbox_delta, self.lat + self.bbox_delta]
        try:
            search = self.catalog.search(collections=[self.collection], bbox=bbox, datetime=f"{start}T00:00:00Z/{end}T23:59:59Z", query={"eo:cloud_cover": {"lte": self.max_cloud_cover}}, limit=max(1, min(limit, 100)), sortby=[{"field": "datetime", "direction": "desc"}])
            items = list(search.items())
        except Exception as exc:
            raise DataUnavailableError("Microsoft Planetary Computer Sentinel-2 catalog is unavailable.", details={"provider": "microsoft-planetary-computer", "reason": str(exc)}) from exc
        records = []
        for item in items:
            assets = self._signed_assets(item)
            normalized = {key.upper(): href for key, href in assets.items()}
            if any(band not in normalized for band in SENTINEL2_BANDS):
                continue
            scl = next((href for key, href in assets.items() if key.upper() in {"SCL", "SCL_20M"}), None)
            if scl:
                normalized["SCL"] = scl
            records.append({"site_id": site_id, "scene_id": item.id, "datetime": item.datetime.isoformat() if item.datetime else item.properties.get("datetime"), "cloud_cover_pct": item.properties.get("eo:cloud_cover"), "collection": self.collection, "geometry": item.geometry, "assets": normalized, "source_uri": item.self_href or self.stac_url, "provider": "microsoft-planetary-computer"})
        acquired_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()
        provenance = DataProvenance(source_name="Copernicus Sentinel-2 L2A via Microsoft Planetary Computer", source_kind="remote_api", mode="live", dataset="sentinel2_scenes", acquired_at=acquired_at, ingested_at=acquired_at, source_version=self.collection, source_uri=self.stac_url, checksum=checksum, license_note="Sentinel-2 L2A public data accessed through Microsoft Planetary Computer; verify source terms for production deployment.", quality_score=1.0 if records else 0.0, row_count=len(records))
        return DataBatch(records=records, provenance=provenance)


class PlanetaryComputerLandsatSurfaceTemperatureProvider(_PlanetaryComputerBase):
    """Discover Landsat ST scenes; pixel-level QA decides thermal usability."""

    def __init__(self, *, max_items: int = 50, max_cloud_cover: float | None = None) -> None:
        super().__init__()
        settings = get_settings()
        self.collection = LANDSAT_COLLECTION
        self.max_items = max(1, min(max_items, 100))
        self.max_cloud_cover = settings.landsat_max_cloud_cover if max_cloud_cover is None else max_cloud_cover
        self.bbox_delta = settings.sentinel2_bbox_delta * 2.5

    def discover(self, latitude: float, longitude: float, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        start = start_date or "1900-01-01"
        end = end_date or "2100-01-01"
        bbox = [longitude - self.bbox_delta, latitude - self.bbox_delta, longitude + self.bbox_delta, latitude + self.bbox_delta]
        try:
            search = self.catalog.search(collections=[self.collection], bbox=bbox, datetime=f"{start}T00:00:00Z/{end}T23:59:59Z", limit=self.max_items, sortby=[{"field": "datetime", "direction": "desc"}])
            items = list(search.items())
        except Exception as exc:
            raise DataUnavailableError("Microsoft Planetary Computer Landsat catalog is unavailable.", details={"provider": "microsoft-planetary-computer", "reason": str(exc)}) from exc

        records = []
        for item in items:
            assets = self._signed_assets(item)
            by_lower = {key.lower(): href for key, href in assets.items()}
            thermal = by_lower.get("lwir11") or by_lower.get("lwir") or by_lower.get("st_b10") or by_lower.get("st_b10.tif")
            if not thermal:
                continue
            normalized = dict(assets)
            normalized["surface_temperature"] = thermal
            # Planetary Computer's landsat-c2-l2 collection names the ST QA
            # asset `qa`; it is ST_QA, not a generic scene QA band.
            for canonical, aliases in {
                "st_qa": ("qa", "st_qa", "st_qa.tif"),
                "qa_pixel": ("qa_pixel", "qa_pixel.tif"),
                "qa_radsat": ("qa_radsat", "qa_radsat.tif"),
            }.items():
                for alias in aliases:
                    if alias in by_lower:
                        normalized[canonical] = by_lower[alias]
                        break
            records.append({"scene_id": item.id, "datetime": item.datetime.isoformat() if item.datetime else item.properties.get("datetime"), "cloud_cover_pct": item.properties.get("eo:cloud_cover"), "assets": normalized, "source_uri": item.self_href or self.stac_url, "provider": "microsoft-planetary-computer"})
        return records
