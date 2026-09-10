# Phase 10 — Real Satellite → Reserve AI

Phase 10 connects real Sentinel-2 + Landsat surface-temperature data to the existing Reserve Intelligence models.

## Live data source

MANGAI uses the **Microsoft Planetary Computer STAC API** for live satellite discovery and signed HTTPS raster assets. This avoids direct CDSE S3 credential management while keeping the data path live and reproducible. Planetary Computer's STAC API is public; hosted raster assets are exposed through signed URLs. See the official Microsoft Planetary Computer documentation for the access model.

## Live flow

```text
Microsoft Planetary Computer STAC
        ↓
Sentinel-2 L2A scene discovery
        ↓
Signed HTTPS COG assets
        ↓
AOI clipping + SCL masking
        ↓
Spectral features / indices
        ↓
Landsat Collection 2 Level-2 ST
        ↓
Temporal + spatial fusion
        ↓
Live satellite DataBatch + provenance
        ↓
Nearest geological context join (WGS84 distance)
        ↓
Reserve prospectivity ensemble
        ↓
Grade + thickness regression
        ↓
Conformal intervals
        ↓
Prototype resource potential (P10/P50/P90)
```

## Python dependencies

Install the Planetary Computer client libraries in the active MANGAI environment:

```powershell
python -m pip install -e .
```

The project declares `pystac-client` and `planetary-computer` as runtime dependencies.

## Live smoke test

Required environment variables:

- `SENTINEL2_LATITUDE`
- `SENTINEL2_LONGITUDE`

Optional:

- `SENTINEL2_MAX_CLOUD_COVER`
- `LIVE_SMOKE_START`
- `LIVE_SMOKE_END`
- `LIVE_SMOKE_SITE_ID`

Run:

```powershell
python scripts/live_satellite_smoke.py
```

No `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, or CDSE S3 credentials are required for this Planetary Computer path.

## Geological input boundary

The live serving path intentionally does **not** use synthetic geological data. It expects an operator-supplied file at:

`data/raw/geological.csv`

Required context columns:

- `sample_id`
- `latitude`
- `longitude`
- `elevation_m`
- `slope_deg`
- `aspect_deg`
- `depth_m`
- `formation`

Target columns such as `mn_pct`, `ore_thickness_m`, and `is_manganese` are not required and are deliberately excluded from the serving join to prevent target leakage.

## API

`POST /api/v1/predictions/reserve/live-satellite`

Request fields:

- `site_id`
- `start`
- `end`
- `latitude`
- `longitude`
- `max_temporal_days` (default 16)
- `max_geology_distance_m` (default 500)
- `limit` (default 5)

The endpoint is **live-only**. With `DATA_MODE=demo`, it fails closed instead of silently falling back to synthetic data.

## Scientific boundary

The result is **prototype resource potential**, not an official mineral resource or reserve estimate. The resource calculation retains the existing assumptions and uncertainty treatment and requires validation with real MOIL geological data, domain experts, applicable reporting standards, and operational systems.

## Provenance

The response exposes both geological and fused satellite provenance, including source kind, mode, checksum, source URI, row count, and acquisition metadata where available.
