from __future__ import annotations

import numpy as np
import pytest

from backend.app.adapters.satellite.sentinel2_pixels import (
    REQUIRED_BANDS,
    PixelExtractionConfig,
    Sentinel2PixelExtractor,
)
from backend.app.core.errors import DataUnavailableError


def test_feature_extraction_requires_crs() -> None:
    arrays = {band: np.ones((1, 1), dtype=np.float32) * 1000 for band in REQUIRED_BANDS}
    mask = np.ones((1, 1), dtype=bool)
    with pytest.raises(DataUnavailableError, match="missing CRS"):
        Sentinel2PixelExtractor._features_from_arrays(
            arrays, mask, {"transform": (1, 0, 0, 0, -1, 0), "crs": None}, "site", "scene"
        )


def test_target_resolution_is_restricted() -> None:
    extractor_cls = Sentinel2PixelExtractor
    with pytest.raises(DataUnavailableError, match="10m or 20m"):
        extractor_cls(PixelExtractionConfig(target_resolution_m=30))
