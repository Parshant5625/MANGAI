from __future__ import annotations

import argparse
import os

from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError, ModelUnavailableError
from backend.app.services.live_reserve import LiveSatelliteReserveService


def _parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="MANGAI live reserve inference smoke test")
    parser.add_argument("--site-id", default=os.getenv("MANGAI_SMOKE_SITE_ID", "smoke-test-site"))
    parser.add_argument("--start", default=os.getenv("MANGAI_SMOKE_START", "2026-08-12"))
    parser.add_argument("--end", default=os.getenv("MANGAI_SMOKE_END", "2026-09-11"))
    parser.add_argument("--latitude", type=float, default=float(os.getenv("SENTINEL2_LATITUDE", settings.sentinel2_latitude)))
    parser.add_argument("--longitude", type=float, default=float(os.getenv("SENTINEL2_LONGITUDE", settings.sentinel2_longitude)))
    parser.add_argument("--max-temporal-days", type=int, default=16)
    parser.add_argument("--max-geology-distance-m", type=float, default=500.0)
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = get_settings()
    print("MANGAI live reserve inference smoke test")
    print("provider: Microsoft Planetary Computer")
    print(f"data_mode: {settings.data_mode}")
    print(f"window: {args.start} -> {args.end}")
    print(f"site_id: {args.site_id}")
    print(f"coordinates: {args.latitude}, {args.longitude}")
    print(f"max geology match distance: {args.max_geology_distance_m} m")

    if settings.data_mode != "live":
        print("FAIL: DATA_MODE must be set to live for this smoke test.")
        return 2

    try:
        result = LiveSatelliteReserveService().predict(
            site_id=args.site_id,
            start=args.start,
            end=args.end,
            latitude=args.latitude,
            longitude=args.longitude,
            max_temporal_days=args.max_temporal_days,
            max_geology_distance_m=args.max_geology_distance_m,
            limit=args.limit,
        )
    except (DataUnavailableError, ModelUnavailableError) as exc:
        print(f"FAIL: {exc}")
        if getattr(exc, "details", None):
            print(f"details: {exc.details}")
        return 1
    except Exception as exc:  # pragma: no cover - smoke-test diagnostic guard
        print(f"FAIL: unexpected live reserve inference error: {type(exc).__name__}: {exc}")
        return 1

    cells = result["cells"]
    probabilities = [float(cell["probability"]) for cell in cells]
    grades = [float(cell["predicted_grade_pct"]) for cell in cells]
    thicknesses = [float(cell["predicted_thickness_m"]) for cell in cells]
    tonnages = [float(cell["resource_potential"]["p50"]) for cell in cells]

    print(f"PASS: matched geological context rows: {result['matched_geological_context']}")
    print(f"unmatched satellite pixels: {result['unmatched_satellite']}")
    print(f"Sentinel-2 scenes: {result['sentinel_scene_count']}")
    print(f"Landsat thermal scenes discovered: {result['thermal_scene_count']}")
    print(f"temporal distances (days): {result['temporal_distance_days']}")
    print(f"PASS: live inference cells: {result['count']}")
    print(f"probability range: {min(probabilities):.4f} -> {max(probabilities):.4f}")
    print(f"grade range (% Mn): {min(grades):.2f} -> {max(grades):.2f}")
    print(f"thickness range (m): {min(thicknesses):.2f} -> {max(thicknesses):.2f}")
    print(f"P50 prototype resource potential (t): {sum(tonnages):,.2f}")
    print(f"boundary: {result['boundary_notice']}")
    print(f"satellite source: {result['satellite_provenance']['source_name']}")
    print(f"satellite checksum: {result['satellite_provenance']['checksum']}")
    print(f"geology checksum: {result['geology_provenance']['checksum']}")
    print("PASS: real satellite + geological context reached reserve inference.")
    print("NOTE: reserve models must be field-trained/validated before operational or regulatory use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
