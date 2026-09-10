# Phase 10 — Real Satellite → Reserve AI

Phase 10 connects the real Sentinel-2 + Landsat surface-temperature fusion pipeline to the existing Reserve Intelligence models.

## Live flow

```text
Sentinel-2 L2A scenes
        ↓
AOI clipping + SCL masking
        ↓
Spectral features / indices
        ↓
Landsat Collection 2 surface temperature
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
