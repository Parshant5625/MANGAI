from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError
from ml.common.provenance import DataBatch, DataProvenance


class Sentinel2STACProvider:
    """Discover real Sentinel-2 L2A scenes through the Copernicus STAC API.

    This adapter deliberately returns scene metadata and asset references rather
    than fabricating pixel values. Pixel extraction/feature generation is a
    separate ingestion step so that MANGAI never presents catalog metadata as
    measured spectral observations.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.catalog_url = settings.sentinel2_stac_url.rstrip("/")
        self.collection = settings.sentinel2_collection
        self.max_cloud_cover = settings.sentinel2_max_cloud_cover
        self.timeout_seconds = settings.sentinel2_timeout_seconds
        self.lat = settings.sentinel2_latitude
        self.lon = settings.sentinel2_longitude
        self.bbox_delta = settings.sentinel2_bbox_delta
        if self.lat is None or self.lon is None:
            raise DataUnavailableError(
                "Live Sentinel-2 requires SENTINEL2_LATITUDE and SENTINEL2_LONGITUDE.",
                details={"provider": "copernicus-stac", "configuration": "missing_coordinates"},
            )

    def search_scenes(self, site_id: str, start: str, end: str, limit: int = 10) -> DataBatch:
        if not start.strip() or not end.strip():
            raise DataUnavailableError("Sentinel-2 search requires a start and end date.")
        if start > end:
            raise DataUnavailableError("Sentinel-2 end date must not precede start date.")

        delta = self.bbox_delta
        bbox = [self.lon - delta, self.lat - delta, self.lon + delta, self.lat + delta]
        payload = {
            "collections": [self.collection],
            "bbox": bbox,
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": max(1, min(limit, 100)),
            "query": {"eo:cloud_cover": {"lte": self.max_cloud_cover}},
            "sortby": [{"field": "datetime", "direction": "desc"}],
        }
        request = Request(
            f"{self.catalog_url}/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Accept": "application/geo+json, application/json", "Content-Type": "application/json", "User-Agent": "MANGAI/1.0"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                response_payload = json.loads(raw)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise DataUnavailableError(
                "Copernicus Sentinel-2 catalog is unavailable.",
                details={"provider": "copernicus-stac", "reason": str(exc)},
            ) from exc

        features = response_payload.get("features", [])
        records: list[dict] = []
        for feature in features:
            props = feature.get("properties") or {}
            assets = feature.get("assets") or {}
            records.append(
                {
                    "site_id": site_id,
                    "scene_id": feature.get("id"),
                    "datetime": props.get("datetime"),
                    "cloud_cover_pct": props.get("eo:cloud_cover"),
                    "collection": feature.get("collection", self.collection),
                    "geometry": feature.get("geometry"),
                    "assets": {
                        key: asset.get("href")
                        for key, asset in assets.items()
                        if isinstance(asset, dict) and asset.get("href")
                    },
                }
            )
        if not records:
            raise DataUnavailableError(
                "No Sentinel-2 L2A scenes matched the requested window and cloud threshold.",
                details={"provider": "copernicus-stac", "collection": self.collection, "max_cloud_cover": self.max_cloud_cover},
            )

        acquired_at = datetime.now(timezone.utc).isoformat()
        checksum = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
        provenance = DataProvenance(
            source_name="Copernicus Data Space Ecosystem Sentinel-2 STAC",
            source_kind="satellite",
            mode="live",
            dataset="satellite_features",
            acquired_at=acquired_at,
            ingested_at=acquired_at,
            source_version=self.collection,
            source_uri=f"{self.catalog_url}/search",
            checksum=checksum,
            license_note="Copernicus Sentinel data are made available free of charge; verify applicable access and usage terms for deployment.",
            quality_score=max(0.0, min(1.0, 1.0 - (sum(float(r["cloud_cover_pct"] or 0) for r in records) / len(records)) / 100.0)),
            row_count=len(records),
        )
        return DataBatch(records=tuple(records), provenance=provenance)
