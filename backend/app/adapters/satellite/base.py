from typing import Protocol


class SatelliteProvider(Protocol):
    def fetch_observations(self, site_id: str, start: str, end: str) -> list[dict]:
        ...

