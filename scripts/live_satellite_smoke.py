"""Run a fail-closed smoke test against Microsoft Planetary Computer.

This script intentionally does not use demo data and does not train models. It
verifies real Sentinel-2 discovery and that the returned raster assets are
signed HTTPS URLs suitable for Rasterio/HTTP access. Planetary Computer uses
short-lived SAS tokens for its hosted raster assets, so no AWS/CDSE credentials
are required.

Required environment variables:
    SENTINEL2_LATITUDE
    SENTINEL2_LONGITUDE

Optional:
    LIVE_SMOKE_SITE_ID (default: smoke-test-site)
    LIVE_SMOKE_START (default: 30 days before LIVE_SMOKE_END)
    LIVE_SMOKE_END (default: today UTC)
    SENTINEL2_MAX_CLOUD_COVER
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from backend.app.adapters.satellite.planetary_computer import PlanetaryComputerSentinel2Provider
from backend.app.core.errors import DataUnavailableError


def _date_range() -> tuple[str, str]:
    end = os.getenv("LIVE_SMOKE_END", datetime.now(UTC).date().isoformat())
    start = os.getenv(
        "LIVE_SMOKE_START",
        (datetime.fromisoformat(end).date() - timedelta(days=30)).isoformat(),
    )
    return start, end


def main() -> int:
    site_id = os.getenv("LIVE_SMOKE_SITE_ID", "smoke-test-site")
    start, end = _date_range()
    print("MANGAI live satellite smoke test")
    print("provider: Microsoft Planetary Computer")
    print(f"window: {start} -> {end}")
    print(f"site_id: {site_id}")

    try:
        provider = PlanetaryComputerSentinel2Provider()
        batch = provider.search_scenes(site_id, start, end, limit=3)
    except DataUnavailableError as exc:
        print(f"FAIL: Sentinel-2 discovery unavailable: {exc.message}")
        if exc.details:
            print(f"details: {exc.details}")
        return 2

    print(f"PASS: discovered {len(batch.records)} Sentinel-2 scene(s)")
    print(f"quality_score: {batch.provenance.quality_score:.3f}")

    schemes: dict[str, int] = {}
    signed_assets = 0
    for scene in batch.records:
        for href in (scene.get("assets") or {}).values():
            scheme = str(href).split(":", 1)[0].lower() if ":" in str(href) else "relative"
            schemes[scheme] = schemes.get(scheme, 0) + 1
            if str(href).startswith("https://") and "sig=" in str(href):
                signed_assets += 1

    print(f"asset schemes: {schemes}")
    print(f"signed HTTPS assets: {signed_assets}")

    if not signed_assets:
        print("BLOCKED: Planetary Computer returned no signed HTTPS raster assets.")
        return 3

    print("PASS: Planetary Computer discovery and signed raster-access prerequisites are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
