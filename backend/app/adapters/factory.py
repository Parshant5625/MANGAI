from backend.app.adapters.satellite.local_file import LocalFileSatelliteProvider
from backend.app.adapters.weather.local_file import LocalFileWeatherProvider
from backend.app.core.config import get_settings


def satellite_provider():
    return LocalFileSatelliteProvider()


def weather_provider():
    return LocalFileWeatherProvider()


def active_mode() -> str:
    return get_settings().data_mode
