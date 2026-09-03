from backend.app.services.demo_data import DemoDataStore


class LocalFileSatelliteProvider:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()

    def fetch_observations(self, site_id: str, start: str, end: str) -> list[dict]:
        df = self.store.satellite().head(500)
        return df.to_dict(orient="records")
