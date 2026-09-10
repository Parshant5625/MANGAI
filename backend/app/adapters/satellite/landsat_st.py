from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backend.app.core.errors import DataUnavailableError


@dataclass(frozen=True)
class LandsatSTConfig:
    stac_url: str = "https://landsatlook.usgs.gov/stac-server"
    collection: str = "landsat-c2l2-st"
    max_cloud_cover: float = 30.0
    timeout_seconds: float = 20.0
    max_items: int = 5

    def __post_init__(self) -> None:
        if not 0 <= self.max_cloud_cover <= 100:
            raise DataUnavailableError("Landsat cloud threshold must be between 0 and 100.")
        if self.timeout_seconds <= 0 or self.max_items < 1:
            raise DataUnavailableError("Landsat ST request limits must be positive.")


class LandsatSurfaceTemperatureProvider:
    """Discover real Landsat Collection 2 Level-2 ST scenes via USGS STAC."""

    def __init__(self, config: LandsatSTConfig | None = None) -> None:
        self.config = config or LandsatSTConfig()

    def discover(
        self,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise DataUnavailableError("Landsat coordinates are outside valid WGS84 bounds.")

        end = datetime.fromisoformat(end_date).date() if end_date else datetime.now(UTC).date()
        start = datetime.fromisoformat(start_date).date() if start_date else end - timedelta(days=30)
        if start > end:
            raise DataUnavailableError("Landsat start_date must not be after end_date.")

        delta = 0.05
        params = urlencode(
            {
                "bbox": f"{longitude - delta},{latitude - delta},{longitude + delta},{latitude + delta}",
                "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
                "limit": self.config.max_items,
            }
        )
        url = f"{self.config.stac_url.rstrip('/')}/collections/{self.config.collection}/items?{params}"
        payload = self._get_json(url)
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            raise DataUnavailableError("USGS Landsat STAC returned an invalid feature collection.")

        scenes: list[dict[str, Any]] = []
        for item in features:
            properties = item.get("properties") or {}
            cloud = self._cloud_cover(properties)
            if cloud is not None and cloud > self.config.max_cloud_cover:
                continue
            assets = item.get("assets") or {}
            st_asset = self._select_st_asset(assets)
            if not st_asset:
                continue
            scenes.append(
                {
                    "scene_id": item.get("id"),
                    "datetime": item.get("datetime"),
                    "collection": self.config.collection,
                    "cloud_cover": cloud,
                    "assets": {"surface_temperature": st_asset},
                    "source_uri": item.get("self") or url,
                }
            )
        return scenes

    @staticmethod
    def _cloud_cover(properties: dict[str, Any]) -> float | None:
        for key in ("eo:cloud_cover", "cloud_cover", "landsat:cloud_cover_land"):
            value = properties.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _select_st_asset(assets: dict[str, Any]) -> str | None:
        preferred = ("ST", "ST_B10", "surface_temperature", "st")
        for key in preferred:
            asset = assets.get(key)
            if isinstance(asset, dict) and asset.get("href"):
                return str(asset["href"])
            if isinstance(asset, str) and asset:
                return asset
        for key, asset in assets.items():
            if "temperature" in key.lower() or key.upper().startswith("ST_"):
                href = asset.get("href") if isinstance(asset, dict) else asset
                if href:
                    return str(href)
        return None

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/geo+json, application/json", "User-Agent": "MANGAI/1.0"})
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise DataUnavailableError(
                "USGS Landsat STAC is unavailable.",
                details={"reason": str(exc)},
            ) from exc
