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
            raise DataUnavailableError(
                "Microsoft Planetary Computer access requires pystac-client and planetary-computer.",
                details={"dependencies": ["pystac-client", "planetary-computer"]},
            )
        self.stac_url = stac_url.rstrip("/")
        try:
            self.catalog = pystac_client.Client.open(
                self.stac_url,
                modifier=planetary_computer.sign_inplace,
            )
        except Exception as exc:
            raise DataUnavailableError(
                "Microsoft Planetary Computer STAC is unavailable.",
                details={"provider": "microsoft-planetary-computer", "reason": str(exc)},
            ) from exc

    @staticmethod
    def _signed_assets(item: Any) -> dict[str, str]:
        assets: dict[str, str] = {}
        for key, asset in item.assets.items():
            if asset.href:
                assets[str(key)] = str(asset.href)
        return assets


class PlanetaryComputerSentinel2Provider(_PlanetaryComputerBase):
    """Discover signed Sentinel-2 L2A assets from Microsoft Planetary Computer."""

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
            raise DataUnavailableError(
                "Live Sentinel-2 requires coordinates.",
                details={"provider": "microsoft-planetary-computer", "configuration": "missing_coordinates"},
            )
        if not start.strip() or not end.strip():
            raise DataUnavailableError("Sentinel-2 search requires a start and end date.")
        if start > end:
            raise DataUnavailableError("Sentinel-2 end date must not precede start date.")

        bbox = [
            self.lon - self.bbox_delta,
            self.lat - self.bbox_delta,
            self.lon + self.bbox_delta,
            self.lat + self.bbox_delta,
        ]
        try:
            search = self.catalog.search(
                collections=[self.collection],
                bbox=bbox,
                datetime=f"{start}T00:00:00Z/{end}T23:59:59Z",
                query={"eo:cloud_cover": {"lte": self.max_cloud_cover}},
                limit=max(1, min(limit, 100)),
                sortby=[{"field": "datetime", "direction": "desc"}],
            )
            items = list(search.items())
        except Exception as exc:
            raise DataUnavailableError(
                "Microsoft Planetary Computer Sentinel-2 catalog is unavailable.",
                details={"provider": "microsoft-planetary-computer", "reason": str(exc)},
            ) from exc

        records: list[dict[str, Any]] = []
        for item in items:
            assets = self._signed_assets(item)
            normalized = {key.upper(): href for key, href in assets.items()}
            missing = [band for band in SENTINEL2_BANDS if band not in normalized]
            if missing:
                continue
            scl = next((href for key, href in assets.items() if key.upper() in {"SCL", "SCL_20M"}), None)
            if scl:
                normalized["SCL"] = scl
            cloud = item.properties.get("eo:cloud_cover")
            records.append(
                {
                    "site_id": site_id,
                    "scene_id": item.id,
                    "datetime": item.datetime.isoformat() if item.datetime else item.properties.get("datetime"),
                    "cloud_cover_pct": cloud,
                    "collection": self.collection,
                    "geometry": item.geometry,
                    "assets": normalized,
                    "source_uri": item.self_href or self.stac_url,
                    "provider": "microsoft-planetary-computer",
                }
            )
        acquired_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()
        provenance = DataProvenance(
            source_name="Copernicus Sentinel-2 L2A via Microsoft Planetary Computer",
            source_kind="remote_api",
            mode="live",
            dataset="sentinel2_scenes",
            acquired_at=acquired_at,
            ingested_at=acquired_at,
            source_version=self.collection,
            source_uri=self.stac_url,
            checksum=checksum,
            license_note="Sentinel-2 L2A public data accessed through Microsoft Planetary Computer; verify source terms for production deployment.",
            quality_score=1.0 if records else 0.0,
            row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)


class PlanetaryComputerLandsatSurfaceTemperatureProvider(_PlanetaryComputerBase):
    """Discover Landsat Collection 2 Level-2 surface-temperature assets.

    Scene-level cloud cover is retained as metadata rather than used as a hard
    acceptance gate. Thermal usability must be decided from pixel-level QA
    (QA_PIXEL/ST_QA/QA_RADSAT) during fusion because a cloudy scene can still
    contain valid thermal pixels inside the requested AOI.
    """

    def __init__(self, *, max_items: int = 50, max_cloud_cover: float | None = None) -> None:
        super().__init__()
        settings = get_settings()
        self.collection = LANDSAT_COLLECTION
        self.max_items = max(1, min(max_items, 100))
        self.max_cloud_cover = settings.landsat_max_cloud_cover if max_cloud_cover is None else max_cloud_cover
        self.bbox_delta = settings.sentinel2_bbox_delta * 2.5

    def discover(
        self,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        start = start_date or "1900-01-01"
        end = end_date or "2100-01-01"
        bbox = [
            longitude - self.bbox_delta,
            latitude - self.bbox_delta,
            longitude + self.bbox_delta,
            latitude + self.bbox_delta,
        ]
        try:
            search = self.catalog.search(
                collections=[self.collection],
                bbox=bbox,
                datetime=f"{start}T00:00:00Z/{end}T23:59:59Z",
                limit=self.max_items,
                sortby=[{"field": "datetime", "direction": "desc"}],
            )
            items = list(search.items())
        except Exception as exc:
            raise DataUnavailableError(
                "Microsoft Planetary Computer Landsat catalog is unavailable.",
                details={"provider": "microsoft-planetary-computer", "reason": str(exc)},
            ) from exc

        records: list[dict[str, Any]] = []
        for item in items:
            cloud = item.properties.get("eo:cloud_cover")
            assets = self._signed_assets(item)
            thermal = assets.get("lwir11") or assets.get("lwir") or assets.get("ST_B10")
            if not thermal:
                continue
            records.append(
                {
                    "scene_id": item.id,
                    "datetime": item.datetime.isoformat() if item.datetime else item.properties.get("datetime"),
                    "cloud_cover_pct": cloud,
                    "assets": {**assets, "surface_temperature": thermal},
                    "source_uri": item.self_href or self.stac_url,
                    "provider": "microsoft-planetary-computer",
                }
            )
        return records
