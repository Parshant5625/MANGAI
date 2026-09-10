from __future__ import annotations

import pandas as pd

from ml.common.validation import validate_dataset
from ml.common.contracts import SATELLITE_CONTRACT


def test_real_pixel_output_can_validate_against_canonical_contract() -> None:
    row = {
        "sample_id": "scene_0_0",
        "latitude": 21.0,
        "longitude": 75.0,
        "blue_b2": 0.10,
        "green_b3": 0.12,
        "red_b4": 0.15,
        "nir_b8": 0.30,
        "swir_b11": 0.20,
        "swir_b12": 0.18,
        "ndvi": 0.333332,
        "ndwi": -0.42857,
        "swir_ratio": 1.1111,
        "bare_soil_index": -0.04,
        "land_surface_temperature": 30.0,
    }
    result = validate_dataset(pd.DataFrame([row]), SATELLITE_CONTRACT)
    assert result.errors == []
