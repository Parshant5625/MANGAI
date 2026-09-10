from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from backend.app.core.errors import DataUnavailableError
from backend.app.services.operations import OperationsService


class EmptyWeatherProvider:
    def fetch_forecast(self, site_id: str, start: str, end: str) -> list[dict]:
        return []


def test_open_meteo_records_are_normalized(monkeypatch) -> None:
    from backend.app.adapters.weather.open_meteo import OpenMeteoWeatherProvider

    settings = SimpleNamespace(
        weather_latitude=21.8,
        weather_longitude=80.0,
        weather_forecast_url="https://example.test/forecast",
        weather_archive_url="https://example.test/archive",
        weather_timeout_seconds=5.0,
    )
    monkeypatch.setattr("backend.app.adapters.weather.open_meteo.get_settings", lambda: settings)
    provider = OpenMeteoWeatherProvider()
    payload = {
        "daily": {
            "time": ["2026-09-01", "2026-09-02"],
            "precipitation_sum": [4.2, 0.0],
            "temperature_2m_mean": [28.1, 29.4],
            "soil_moisture_0_to_7cm_mean": [0.31, 0.28],
        }
    }
    records = provider._records(payload, date(2026, 9, 1), date(2026, 9, 2))
    assert records == [
        {"date": "2026-09-01", "rainfall_mm": 4.2, "temperature_c": 28.1, "soil_moisture": 0.31},
        {"date": "2026-09-02", "rainfall_mm": 0.0, "temperature_c": 29.4, "soil_moisture": 0.28},
    ]


def test_live_provider_requires_coordinates(monkeypatch) -> None:
    from backend.app.adapters.weather.open_meteo import OpenMeteoWeatherProvider

    settings = SimpleNamespace(
        weather_latitude=None,
        weather_longitude=None,
        weather_forecast_url="https://example.test/forecast",
        weather_archive_url="https://example.test/archive",
        weather_timeout_seconds=5.0,
    )
    monkeypatch.setattr("backend.app.adapters.weather.open_meteo.get_settings", lambda: settings)
    with pytest.raises(DataUnavailableError, match="WEATHER_LATITUDE"):
        OpenMeteoWeatherProvider()


def test_live_weather_does_not_fallback_to_demo(monkeypatch) -> None:
    settings = SimpleNamespace(data_mode="live", demo_site_id="demo-moil-site")
    monkeypatch.setattr("backend.app.services.operations.get_settings", lambda: settings)
    monkeypatch.setattr("backend.app.services.operations.weather_provider", lambda: EmptyWeatherProvider())
    with pytest.raises(ValueError, match="no observations"):
        OperationsService().weather()
