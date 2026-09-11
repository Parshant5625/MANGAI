from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend.app.adapters.satellite.landsat_st_fusion import (
    ST_OFFSET_K,
    ST_SCALE,
    LandsatSurfaceTemperatureFusion,
)
from ml.common.provenance import DataBatch, DataProvenance


def _batch() -> DataBatch:
    records = [
        {"sample_id": "s1", "x": 0.5, "y": 9.5, "crs": "EPSG:4326", "blue_b2": 0.2, "land_surface_temperature": None},
        {"sample_id": "s2", "x": 1.5, "y": 8.5, "crs": "EPSG:4326", "blue_b2": 0.3, "land_surface_temperature": None},
    ]
    return DataBatch(records=records, provenance=DataProvenance(source_name="Sentinel-2", source_kind="satellite", mode="live", dataset="satellite_features", row_count=len(records)))


def _raster_bytes(values: np.ndarray) -> bytes:
    output = BytesIO()
    with MemoryFile() as memory_file:
        with memory_file.open(driver="GTiff", height=values.shape[0], width=values.shape[1], count=1, dtype="uint16", crs="EPSG:4326", transform=from_origin(0, 10, 1, 1)) as dataset:
            dataset.write(values, 1)
        output.write(memory_file.read())
    return output.getvalue()


def test_fuses_scaled_landsat_temperature(monkeypatch):
    dn = np.array([[30000, 0], [31000, 32000]], dtype=np.uint16)
    raw = _raster_bytes(dn)
    fusion = LandsatSurfaceTemperatureFusion()
    monkeypatch.setattr(fusion, "_download", lambda href: raw)
    result = fusion.fuse(_batch(), {"scene_id": "LC09_TEST", "datetime": "2026-07-30T00:00:00Z", "collection": "landsat-c2l2-st", "assets": {"surface_temperature": "https://example.test/st.tif"}})
    assert len(result.records) == 2
    expected = 30000 * ST_SCALE + ST_OFFSET_K - 273.15
    assert result.records[0]["land_surface_temperature"] == pytest.approx(expected, abs=1e-5)
    assert result.records[0]["thermal_scene_id"] == "LC09_TEST"
    assert result.provenance.quality_score == 1.0


def test_drops_fill_pixels(monkeypatch):
    raw = _raster_bytes(np.array([[30000, 0], [0, 0]], dtype=np.uint16))
    fusion = LandsatSurfaceTemperatureFusion()
    monkeypatch.setattr(fusion, "_download", lambda href: raw)
    result = fusion.fuse(_batch(), {"scene_id": "LC09_TEST", "assets": {"surface_temperature": "x"}})
    assert len(result.records) == 1
    assert result.records[0]["sample_id"] == "s1"
    assert result.provenance.quality_score == 0.5


def test_masked_uint16_array_accepts_nan_safely():
    # Regression for: TypeError: Cannot convert fill_value nan to dtype uint16
    array = np.ma.array(np.array([[0, 30000], [0, 0]], dtype=np.uint16), mask=np.array([[True, False], [True, True]]))
    value = LandsatSurfaceTemperatureFusion._nearest_valid_from_array(array, 0, 0)
    assert value == pytest.approx(30000.0)


def test_rejects_demo_batch():
    batch = _batch()
    demo = DataBatch(records=batch.records, provenance=DataProvenance(source_name="demo", source_kind="local_file", mode="demo", dataset="satellite_features", row_count=len(batch.records)))
    fusion = LandsatSurfaceTemperatureFusion()
    try:
        fusion.fuse(demo, {"assets": {"surface_temperature": "x"}})
    except Exception as exc:
        assert "live" in str(exc).lower()
    else:
        raise AssertionError("demo thermal fusion must be rejected")
