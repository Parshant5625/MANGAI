from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backend.app.core.config import get_settings
from backend.app.core.errors import DataUnavailableError


class OpenMeteoWeatherProvider:
    """Live weather adapter backed by Open-Meteo forecast/archive APIs."""

    def __init__(self) -> None:
        settings = get_settings()
        self.latitude = settings.weather_latitude
        self.longitude = settings.weather_longitude
        self.forecast_url = settings.weather_forecast_url
        self.archive_url = settings.weather_archive_url
        self.timeout_seconds = settings.weather_timeout_seconds
        if self.latitude is None or self.longitude is None:
            raise DataUnavailableError(
                "Live weather requires WEATHER_LATITUDE and WEATHER_LONGITUDE.",
                details={"provider": "open-meteo", "configuration": "missing_coordinates"},
            )

    def fetch_forecast(self, site_id: str, start: str, end: str) -> list[dict]:
        start_date, end_date = self._resolve_window(start, end)
        today = datetime.now(timezone.utc).date()
        base_url = self.archive_url if end_date < today else self.forecast_url
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "precipitation_sum,temperature_2m_mean,soil_moisture_0_to_7cm_mean",
            "timezone": "UTC",
        }
        if base_url == self.forecast_url:
            params["past_days"] = max(1, min(92, (today - start_date).days))
            params.pop("start_date", None)
            params.pop("end_date", None)
            params["forecast_days"] = max(1, min(16, (end_date - today).days + 1))

        request_url = f"{base_url}?{urlencode(params)}"
        request = Request(request_url, headers={"Accept": "application/json", "User-Agent": "MANGAI/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise DataUnavailableError(
                "Live weather provider is unavailable.",
                details={"provider": "open-meteo", "reason": str(exc)},
            ) from exc
        return self._records(payload, start_date, end_date)

    @staticmethod
    def _resolve_window(start: str, end: str) -> tuple[date, date]:
        today = datetime.now(timezone.utc).date()
        if not start and not end:
            return today - timedelta(days=29), today
        start_date = date.fromisoformat(start) if start else today
        end_date = date.fromisoformat(end) if end else start_date
        if end_date < start_date:
            raise DataUnavailableError("Weather end date must not precede start date.")
        return start_date, end_date

    @staticmethod
    def _records(payload: dict, start_date: date, end_date: date) -> list[dict]:
        daily = payload.get("daily")
        if not isinstance(daily, dict) or "time" not in daily:
            raise DataUnavailableError(
                "Live weather response did not contain daily observations.",
                details={"provider": "open-meteo"},
            )
        records: list[dict] = []
        times = daily.get("time", [])
        rainfall = daily.get("precipitation_sum", [])
        temperatures = daily.get("temperature_2m_mean", [])
        soil = daily.get("soil_moisture_0_to_7cm_mean", [])
        for index, value in enumerate(times):
            current = date.fromisoformat(str(value))
            if not start_date <= current <= end_date:
                continue
            records.append(
                {
                    "date": current.isoformat(),
                    "rainfall_mm": rainfall[index] if index < len(rainfall) else None,
                    "temperature_c": temperatures[index] if index < len(temperatures) else None,
                    "soil_moisture": soil[index] if index < len(soil) else None,
                }
            )
        if not records:
            raise DataUnavailableError(
                "Live weather provider returned no observations for the requested window.",
                details={"provider": "open-meteo", "start": start_date.isoformat(), "end": end_date.isoformat()},
            )
        return records
