from backend.app.adapters.satellite.local_file import LocalFileSatelliteProvider
from backend.app.adapters.satellite.planetary_computer import PlanetaryComputerSentinel2Provider
from backend.app.adapters.weather.local_file import LocalFileWeatherProvider
from backend.app.adapters.weather.open_meteo import OpenMeteoWeatherProvider
from backend.app.core.config import get_settings


def satellite_provider():
    settings = get_settings()
    if settings.data_mode == "live":
        return PlanetaryComputerSentinel2Provider()
    return LocalFileSatelliteProvider()


def weather_provider():
    settings = get_settings()
    if settings.data_mode == "live":
        return OpenMeteoWeatherProvider()
    return LocalFileWeatherProvider()


def active_mode() -> str:
    return get_settings().data_mode
