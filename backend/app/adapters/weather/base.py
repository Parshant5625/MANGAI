from typing import Protocol


class WeatherProvider(Protocol):
    def fetch_forecast(self, site_id: str, start: str, end: str) -> list[dict]:
        ...

