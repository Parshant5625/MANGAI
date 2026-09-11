from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError, ModelUnavailableError
from backend.app.services.live_reserve import LiveSatelliteReserveService


def _coordinate_default(env_name: str, configured: float | None, fallback: float) -> float:
    raw = os.getenv(env_name)
    if raw is not None and raw.strip():
        return float(raw)
    if configured is not None:
        return float(configured)
    return fallback


def _parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="MANGAI live reserve inference smoke test")
    parser.add_argument("--site-id", default=os.getenv("MANGAI_SMOKE_SITE_ID", "smoke-test-site"))
    parser.add_argument("--start", default=os.getenv("MANGAI_SMOKE_START", "2026-08-12"))
    parser.add_argument("--end", default=os.getenv("MANGAI_SMOKE_END", "2026-09-11"))
    parser.add_argument("--latitude", type=float, default=_coordinate_default("SENTINEL2_LATITUDE", settings.sentinel2_latitude, 21.81))
    parser.add_argument("--longitude", type=float, default=_coordinate_default("SENTINEL2_LONGITUDE", settings.sentinel2_longitude, 80.18))
    parser.add_argument("--max-temporal-days", type=int, default=32)
    parser.add_argument("--max-geology-distance-m", type=float, default=500.0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--demo-geology", action="store_true")
    return parser


def _nearest_demo_coordinate(path: Path, latitude: float, longitude: float) -> tuple[float, float]:
    best: tuple[float, float, float] | None = None
    lon_scale = max(0.1, abs(math.cos(math.radians(latitude))))
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row_lat = float(row["latitude"])
                row_lon = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            distance_sq = (row_lat - latitude) ** 2 + ((row_lon - longitude) * lon_scale) ** 2
            if best is None or distance_sq < best[0]:
                best = (distance_sq, row_lat, row_lon)
    if best is None:
        raise ValueError(f"Synthetic geological context has no valid coordinates: {path}")
    return best[1], best[2]


def main() -> int:
    args = _parser().parse_args()
    settings = get_settings()
    raw_geology = settings.resolved_data_dir / "raw" / "geological.csv"
    synthetic_geology = settings.resolved_data_dir / "synthetic" / "geological.csv"
    geology_path: Path | None = None

    if args.demo_geology:
        geology_path = synthetic_geology
        if not geology_path.exists():
            print(f"FAIL: synthetic geological context is unavailable: {geology_path}")
            return 2
        original_coordinates = (args.latitude, args.longitude)
        args.latitude, args.longitude = _nearest_demo_coordinate(geology_path, args.latitude, args.longitude)
        if args.max_geology_distance_m == 500.0:
            args.max_geology_distance_m = 2500.0
        print("WARNING: DEMO MIXED-DATA MODE — real satellite + synthetic geological context.")
        print(f"synthetic geology: {geology_path}")
        print(f"demo AOI aligned to nearest synthetic geology coordinate: {args.latitude}, {args.longitude}")
        print(f"note: requested satellite coordinates were {original_coordinates[0]}, {original_coordinates[1]}")
    else:
        geology_path = raw_geology

    print("MANGAI live reserve inference smoke test")
    print("provider: Microsoft Planetary Computer")
    print(f"data_mode: {settings.data_mode}")
    print(f"window: {args.start} -> {args.end}")
    print(f"site_id: {args.site_id}")
    print(f"coordinates: {args.latitude}, {args.longitude}")
    print(f"max temporal distance: {args.max_temporal_days} days")
    print(f"max geology match distance: {args.max_geology_distance_m} m")

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
            geology_path=geology_path,
        )
    except (DataUnavailableError, ModelUnavailableError, ValueError) as exc:
        print(f"FAIL: {exc}")
        details = getattr(exc, "details", None)
        if details:
            print(f"details: {details}")
        return 2

    satellite = result["satellite_provenance"]
    geology = result["geology_provenance"]
    print(f"PASS: matched geological context rows: {result['matched_geological_context']}")
    print(f"unmatched satellite pixels: {result['unmatched_satellite']}")
    print(f"Sentinel-2 scenes: {result['sentinel_scene_count']}")
    print(f"Landsat thermal scenes discovered: {result['thermal_scene_count']}")
    print(f"temporal distances (days): {result['temporal_distance_days']}")
    print(f"PASS: live inference cells: {len(result['cells'])}")
    if result["cells"]:
        probabilities = [float(cell["probability"]) for cell in result["cells"]]
        grades = [float(cell["predicted_grade_pct"]) for cell in result["cells"]]
        thickness = [float(cell["predicted_thickness_m"]) for cell in result["cells"]]
        p50 = sum(float(cell["resource_potential"]["p50"]) for cell in result["cells"])
        print(f"probability range: {min(probabilities):.4f} -> {max(probabilities):.4f}")
        print(f"grade range (% Mn): {min(grades):.2f} -> {max(grades):.2f}")
        print(f"thickness range (m): {min(thickness):.2f} -> {max(thickness):.2f}")
        print(f"P50 prototype resource potential (t): {p50:,.2f}")
    print(f"boundary: {result['boundary_notice']}")
    print(f"satellite source: {satellite['source_name']}")
    print(f"satellite checksum: {satellite['checksum']}")
    print(f"geology source: {geology['source_name']}")
    print(f"geology mode: {geology['mode']}")
    print("PASS: real satellite + geological context reached the reserve inference path.")
    if geology["mode"] == "demo":
        print("BOUNDARY: this is a demonstration only; replace synthetic geology with operator-supplied geological data for real inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
