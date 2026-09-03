from __future__ import annotations

import pandas as pd

from backend.app.services.demo_data import DemoDataStore


class LocalFileWeatherProvider:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()

    def fetch_forecast(self, site_id: str, start: str, end: str) -> list[dict]:
        df = self.store.weather().copy()
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]
        records = df.tail(90).copy()
        records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
        return records.to_dict(orient="records")
