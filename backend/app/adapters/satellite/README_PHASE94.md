# Phase 9.4 Sentinel-2 pixel ingestion

This phase reads real Sentinel-2 L2A raster assets discovered through Copernicus STAC and derives B02/B03/B04/B08/B11/B12 reflectance features plus NDVI, NDWI, SWIR ratio and bare-soil index.

Sentinel-2 does not provide thermal land-surface temperature. `land_surface_temperature` therefore remains a separate-source field and must not be fabricated by this adapter.

The canonical model contract currently contains that field as part of the fused satellite dataset. Live Sentinel-2-only batches may therefore require an explicit later fusion step with thermal observations before they are eligible for reserve-model inference.

Target resolution is 20 m by default so the 20 m SWIR bands are not upsampled into falsely precise 10 m detail. Cloud/shadow/snow masking uses the Sentinel-2 Scene Classification Layer when that STAC asset is supplied.
