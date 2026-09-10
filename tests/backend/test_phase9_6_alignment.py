from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from backend.app.adapters.satellite.alignment import select_temporally_matched_scene
from backend.app.adapters.satellite.sentinel2_pixels import PixelExtractionConfig
from backend.app.core.errors import DataUnavailableError


def test_selects_closest_scene_within_window() -> None:
    selected = select_temporally_matched_scene(
        "2026-08-10T00:00:00Z",
        [
            {"scene_id": "far", "datetime": "2026-08-20T00:00:00Z"},
            {"scene_id": "closest", "datetime": "2026-08-11T00:00:00Z"},
            {"scene_id": "middle", "datetime": "2026-08-07T00:00:00Z"},
        ],
        max_days=5,
    )
    assert selected["scene_id"] == "closest"
    assert selected["temporal_distance_days"] == 1.0


def test_rejects_scene_outside_temporal_window() -> None:
    with pytest.raises(DataUnavailableError, match="temporal matching window"):
        select_temporally_matched_scene(
            datetime(2026, 8, 10, tzinfo=UTC),
            [{"scene_id": "far", "datetime": "2026-08-20T00:00:00Z"}],
            max_days=5,
        )


def test_aoi_bbox_is_validated_in_wgs84() -> None:
    config = PixelExtractionConfig(aoi_bbox=(76.0, 20.0, 76.1, 20.1))
    assert config.aoi_bbox == (76.0, 20.0, 76.1, 20.1)


def test_invalid_aoi_bbox_is_rejected() -> None:
    with pytest.raises(DataUnavailableError, match="AOI bbox"):
        PixelExtractionConfig(aoi_bbox=(76.1, 20.0, 76.0, 20.1))


def test_scl_cloud_classes_are_defined_as_masked() -> None:
    from backend.app.adapters.satellite.sentinel2_pixels import SCL_MASK_CLASSES

    assert set(SCL_MASK_CLASSES) == {3, 8, 9, 10, 11}
    assert np.array([3, 8, 9, 10, 11]).size == len(SCL_MASK_CLASSES)
