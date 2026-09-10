# Phase 9.6 — AOI and Spatial/Temporal Alignment

Phase 9.6 makes the real satellite ingestion path explicit about geographic and temporal alignment.

## Spatial handling

- Sentinel-2 spectral bands are reprojected onto one common reference grid.
- The default working resolution remains 20 m because the reserve feature set includes 20 m SWIR bands.
- `PixelExtractionConfig.aoi_bbox` accepts `(min_lon, min_lat, max_lon, max_lat)` in EPSG:4326.
- The AOI is transformed into the raster CRS and clipped before feature rows are emitted.
- Sentinel-2 SCL classes 3, 8, 9, 10 and 11 are masked when an SCL asset is available.
- SCL is resampled with nearest-neighbour; continuous reflectance bands use bilinear resampling.

## Temporal handling

`select_temporally_matched_scene()` selects the closest thermal scene to a Sentinel acquisition timestamp and rejects scenes outside the configured maximum day difference. Invalid or missing timestamps are not silently accepted.

## Data integrity

This phase does not fabricate missing thermal observations. The final Sentinel-2 + Landsat batch remains live and provenance-bearing. A scene with no valid pixels after AOI/quality masking is rejected rather than converted into a synthetic result.

The current Landsat ST downloader remains intentionally simple. Production deployment should replace whole-file downloads with COG window/range reads and add Landsat `QA_PIXEL`/`ST_QA` quality masking.
