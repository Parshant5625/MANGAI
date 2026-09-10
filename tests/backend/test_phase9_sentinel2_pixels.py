from __future__ import annotations

import numpy as np
import pytest

from backend.app.adapters.satellite.sentinel2_pixels import (
    REQUIRED_BANDS,
    PixelExtractionConfig,
    Sentinel2PixelExtractor,
)
from backend.app.core.errors import DataUnavailableError


def test_select_assets_accepts_canonical_and_resolution_suffixes() -> None:
    assets = {f"{band}_20m": f"https://example.test/{band}.tif" for band in REQUIRED_BANDS}
    selected = Sentinel2PixelExtractor._select_assets(assets)
    assert set(selected) == set(REQUIRED_BANDS)


def test_select_assets_requires_all_spectral_bands() -> None:
    assets = {"B02": "a", "B03": "b"}
    with pytest.raises(DataUnavailableError, match="missing required spectral assets"):
        Sentinel2PixelExtractor._select_assets(assets)


def test_feature_math_uses_reflectance_and_masks_invalid_pixels() -> None:
    shape = (2, 2)
    arrays = {
        band: np.full(shape, 1000.0, dtype=np.float32)
        for band in REQUIRED_BANDS
    }
    arrays["B08"][0, 0] = 2000.0
    arrays["B04"][0, 0] = 1000.0
    arrays["B12"][1, 1] = -1.0
    mask = np.array([[True, True], [True, False]])
    profile = {
        "transform": __import__("rasterio").transform.from_origin(75.0, 21.0, 0.01, 0.01),
        "crs": "EPSG:4326",
    }
    rows = Sentinel2PixelExtractor._features_from_arrays(
        arrays, mask, profile, "demo-site", "scene-1"
    )
    assert len(rows) == 3
    first = rows[0]
    assert first["sample_id"] == "scene-1_0_0"
    assert first["nir_b8"] == pytest.approx(0.2)
    assert first["red_b4"] == pytest.approx(0.1)
    assert first["ndvi"] == pytest.approx(1 / 3, rel=1e-5)
    assert first["land_surface_temperature"] is None
    assert first["crs"] == "EPSG:4326"
    assert first["longitude"] == pytest.approx(75.005)
    assert first["latitude"] == pytest.approx(20.995)


def test_extractor_requires_rasterio(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app.adapters.satellite.sentinel2_pixels as module

    monkeypatch.setattr(module, "rasterio", None)
    with pytest.raises(DataUnavailableError, match="rasterio"):
        Sentinel2PixelExtractor(PixelExtractionConfig())
