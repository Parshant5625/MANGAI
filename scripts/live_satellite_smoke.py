"""Run a fail-closed smoke test against the live satellite discovery stack.

This script intentionally does not use demo data and does not train models.
It verifies external catalog discovery first and reports whether raster access is
ready. Copernicus STAC currently commonly returns ``s3://eodata/...`` assets;
reading those assets requires Copernicus S3 credentials in the environment.

Required environment variables:
    SENTINEL2_LATITUDE
    SENTINEL2_LONGITUDE

Optional:
    LIVE_SMOKE_SITE_ID (default: smoke-test-site)
    LIVE_SMOKE_START (default: 30 days before LIVE_SMOKE_END)
    LIVE_SMOKE_END (default: today UTC)
    SENTINEL2_MAX_CLOUD_COVER

The script never prints credential values.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from backend.app.adapters.satellite.sentinel2 import Sentinel2STACProvider
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
    print(f"window: {start} -> {end}")
    print(f"site_id: {site_id}")

    try:
        provider = Sentinel2STACProvider()
        batch = provider.search_scenes(site_id, start, end, limit=3)
    except DataUnavailableError as exc:
        print(f"FAIL: Sentinel-2 discovery unavailable: {exc.message}")
        if exc.details:
            print(f"details: {exc.details}")
        return 2

    print(f"PASS: discovered {len(batch.records)} Sentinel-2 scene(s)")
    print(f"quality_score: {batch.provenance.quality_score:.3f}")

    schemes: dict[str, int] = {}
    for scene in batch.records:
        for href in (scene.get("assets") or {}).values():
            scheme = str(href).split(":", 1)[0].lower() if ":" in str(href) else "relative"
            schemes[scheme] = schemes.get(scheme, 0) + 1
    print(f"asset schemes: {schemes}")

    if "s3" in schemes:
        access_key = bool(os.getenv("AWS_ACCESS_KEY_ID"))
        secret_key = bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
        print(
            "NEXT: Copernicus returned S3 raster assets. "
            "Pixel ingestion requires CDSE S3 credentials."
        )
        print(f"AWS_ACCESS_KEY_ID configured: {access_key}")
        print(f"AWS_SECRET_ACCESS_KEY configured: {secret_key}")
        if not (access_key and secret_key):
            print(
                "BLOCKED: configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "from your Copernicus Data Space S3 credentials, then rerun."
            )
            return 3

    print("PASS: external satellite discovery and raster-access prerequisites are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
