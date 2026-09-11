from backend.app.adapters.satellite.landsat_st_fusion import LandsatSurfaceTemperatureFusion


def test_qa_pixel_flags_are_decoded_individually():
    value = (1 << 0) | (1 << 1) | (1 << 3)
    flags = LandsatSurfaceTemperatureFusion._qa_pixel_flags(value)
    assert flags == ["fill", "dilated_cloud", "cloud"]


def test_clear_qa_pixel_has_no_flags():
    assert LandsatSurfaceTemperatureFusion._qa_pixel_flags(21824) == []


def test_qa_pixel_bad_matches_flag_decoder():
    assert LandsatSurfaceTemperatureFusion._qa_pixel_is_bad(1 << 4)
    assert not LandsatSurfaceTemperatureFusion._qa_pixel_is_bad(21824)
