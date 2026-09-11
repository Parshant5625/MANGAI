from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from backend.app.adapters.satellite.planetary_computer import PlanetaryComputerSentinel2Provider
from backend.app.adapters.satellite.sentinel2_pixel_service import Sentinel2PixelService
from backend.app.adapters.satellite.fusion_pipeline import LIVE_AOI_HALF_DEG
from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError


def main() -> int:
    parser = argparse.ArgumentParser(description="MANGAI live Sentinel-2 pixel-ingestion smoke test")
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--site-id", default="live-pixel-smoke-site")
    args = parser.parse_args()

    settings = get_settings()
    latitude = args.latitude if args.latitude is not None else settings.sentinel2_latitude
    longitude = args.longitude if args.longitude is not None else settings.sentinel2_longitude
    if latitude is None or longitude is None:
        print("BLOCKED: provide SENTINEL2_LATITUDE and SENTINEL2_LONGITUDE or CLI coordinates.")
        return 2

    end = args.end or datetime.now(UTC).date().isoformat()
    start = args.start or (datetime.fromisoformat(end).date() - timedelta(days=30)).isoformat()
    aoi_bbox = (
        max(-180.0, longitude - LIVE_AOI_HALF_DEG),
        max(-90.0, latitude - LIVE_AOI_HALF_DEG),
        min(180.0, longitude + LIVE_AOI_HALF_DEG),
        min(90.0, latitude + LIVE_AOI_HALF_DEG),
    )

    print("MANGAI live Sentinel-2 pixel smoke test")
    print("provider: Microsoft Planetary Computer")
    print(f"window: {start} -> {end}")
    print(f"coordinates: {latitude}, {longitude}")
    print(f"AOI: {aoi_bbox}")
    print(f"cloud threshold: {settings.sentinel2_max_cloud_cover}%")

    try:
        provider = PlanetaryComputerSentinel2Provider()
        batch = provider.search_scenes(
            site_id=args.site_id,
            start=start,
            end=end,
            limit=3,
        )
        scenes = batch.records
        if not scenes:
            print("FAIL: no Sentinel-2 scenes matched the requested window and cloud threshold.")
            return 1

        scene = scenes[0]
        print(f"PASS: discovered {len(scenes)} scene(s); selecting {scene.get('scene_id')}")
        assets = scene.get("assets") or {}
        print(f"spectral assets: {sum(1 for key in ('B02', 'B03', 'B04', 'B08', 'B11', 'B12') if key in assets)} / 6")
        print(f"SCL asset: {'yes' if 'SCL' in assets else 'no'}")

        pixel_batch = Sentinel2PixelService().ingest_scene(
            scene,
            args.site_id,
            aoi_bbox=aoi_bbox,
        )
        print(f"PASS: extracted {len(pixel_batch.records)} valid satellite pixel(s)")
        print(f"quality_score: {pixel_batch.provenance.quality_score}")
        print(f"dataset: {pixel_batch.provenance.dataset}")
        print(f"mode: {pixel_batch.provenance.mode}")
        print(f"source: {pixel_batch.provenance.source_name}")
        if pixel_batch.records:
            sample = pixel_batch.records[0]
            print("sample feature keys:", ", ".join(sorted(sample.keys())))
            print(f"sample WGS84: lat={sample.get('latitude')}, lon={sample.get('longitude')}")
            print(f"sample NDVI: {sample.get('ndvi')}")
            print(f"sample LST: {sample.get('land_surface_temperature')}")
        print("PASS: real Sentinel-2 raster pixels reached the MANGAI satellite feature contract.")
        print("NEXT: run the full live satellite fusion and reserve-inference smoke test.")
        return 0
    except DataUnavailableError as exc:
        print(f"FAIL: {exc}")
        if getattr(exc, "details", None):
            print(f"details: {exc.details}")
        return 1
    except Exception as exc:  # smoke-test diagnostics should remain readable
        print(f"FAIL: unexpected live ingestion error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
