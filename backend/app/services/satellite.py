from __future__ import annotations

from typing import Any

from backend.app.adapters.factory import satellite_provider
from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore


class SatelliteService:
    """Satellite discovery service with explicit demo/live provenance."""

    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.settings = get_settings()
        self.store = store or DemoDataStore()

    def scenes(self, site_id: str | None, start: str, end: str, limit: int = 10) -> dict[str, Any]:
        resolved_site = site_id or self.settings.demo_site_id
        if self.settings.data_mode == "live":
            batch = satellite_provider().search_scenes(resolved_site, start, end, limit=limit)
            return {
                "site_id": resolved_site,
                "mode": "live",
                "records": batch.records,
                "provenance": batch.provenance.to_dict(),
            }

        records = self.store.satellite().head(limit).to_dict(orient="records")
        return {
            "site_id": resolved_site,
            "mode": "demo",
            "records": records,
            "provenance": {
                "source_name": "MANGAI synthetic demo satellite dataset",
                "source_kind": "synthetic",
                "mode": "demo",
                "dataset": "satellite_features",
                "row_count": len(records),
                "is_synthetic": True,
            },
        }
