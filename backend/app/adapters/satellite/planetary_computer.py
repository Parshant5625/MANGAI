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
        if self.lat is None or self.lon is None:
            raise DataUnavailableError(
                "Live Sentinel-2 requires SENTINEL2_LATITUDE and SENTINEL2_LONGITUDE.",
                details={"provider": "microsoft-planetary-computer", "configuration": "missing_coordinates"},
            )

    def search_scenes(self, site_id: str, start: str, end: str, limit: int = 10) -> DataBatch:
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

        if not records:
            raise DataUnavailableError(
                "No Sentinel-2 L2A scenes matched the requested window and cloud threshold.",
                details={
                    "provider": "microsoft-planetary-computer",
                    "collection": self.collection,
                    "max_cloud_cover": self.max_cloud_cover,
                },
            )

        acquired_at = datetime.now(UTC).isoformat()
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
        mean_cloud = sum(float(record["cloud_cover_pct"] or 0) for record in records) / len(records)
        provenance = DataProvenance(
            source_name="Microsoft Planetary Computer Sentinel-2 L2A",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=acquired_at,
            ingested_at=acquired_at,
            source_version=self.collection,
            source_uri=f"{self.stac_url}/search",
            checksum=checksum,
            license_note="Sentinel-2 data are provided by Copernicus/ESA and hosted by Microsoft Planetary Computer; verify applicable terms for deployment.",
            quality_score=max(0.0, min(1.0, 1.0 - mean_cloud / 100.0)),
            row_count=len(records),
        )
        return DataBatch(records=records, provenance=provenance)


class PlanetaryComputerLandsatSurfaceTemperatureProvider(_PlanetaryComputerBase):
    """Discover signed Landsat Collection 2 Level-2 ST assets from Planetary Computer."""

    def __init__(self, max_cloud_cover: float | None = None, max_items: int = 20) -> None:
        super().__init__()
        settings = get_settings()
        configured_cloud = settings.landsat_max_cloud_cover if max_cloud_cover is None else max_cloud_cover
        if not 0 <= configured_cloud <= 100:
            raise DataUnavailableError("Landsat cloud threshold must be between 0 and 100.")
        self.max_cloud_cover = configured_cloud
        self.max_items = max(1, max_items)

    def discover(
        self,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise DataUnavailableError("Landsat coordinates are outside valid WGS84 bounds.")
        end = end_date or datetime.now(UTC).date().isoformat()
        start = start_date or end
        if start > end:
            raise DataUnavailableError("Landsat start_date must not be after end_date.")

        delta = 0.05
        bbox = [longitude - delta, latitude - delta, longitude + delta, latitude + delta]
        try:
            search = self.catalog.search(
                collections=[LANDSAT_COLLECTION],
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

        scenes: list[dict[str, Any]] = []
        for item in items:
            assets = self._signed_assets(item)
            st_asset = assets.get("lwir11") or assets.get("lwir")
            if not st_asset:
                continue
            cloud = item.properties.get("eo:cloud_cover")
            qa_pixel = assets.get("qa_pixel") or assets.get("QA_PIXEL")
            qa_radsat = assets.get("qa_radsat") or assets.get("QA_RADSAT")
            st_qa = assets.get("st_qa") or assets.get("ST_QA")
            scenes.append(
                {
                    "scene_id": item.id,
                    "datetime": item.datetime.isoformat() if item.datetime else item.properties.get("datetime"),
                    "collection": LANDSAT_COLLECTION,
                    "cloud_cover": float(cloud) if cloud is not None else None,
                    "cloud_threshold_pct": self.max_cloud_cover,
                    "cloud_threshold_passed": cloud is None or float(cloud) <= self.max_cloud_cover,
                    "assets": {
                        "surface_temperature": st_asset,
                        "qa_pixel": qa_pixel,
                        "qa_radsat": qa_radsat,
                        "st_qa": st_qa,
                    },
                    "source_uri": item.self_href or self.stac_url,
                    "provider": "microsoft-planetary-computer",
                }
            )
        return scenes
