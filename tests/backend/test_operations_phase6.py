from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.operations_summary import OperationsSummaryService


class FakeStore:
    def __init__(self) -> None:
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        self._production = pd.DataFrame({"date": dates, "production_mt": [100 - i * 0.5 for i in range(30)], "target_mt": [100.0] * 30})
        self._weather = pd.DataFrame({"date": dates, "rainfall_mm": [10.0] * 30, "soil_moisture": [0.2] * 30, "temperature_c": [25.0] * 30})
        self._equipment = pd.DataFrame({"date": dates, "equipment_id": ["E1"] * 30, "downtime_hours": list(range(30)), "utilization": [0.7] * 30, "maintenance": [0] * 30})
        self._blasting = pd.DataFrame({"date": dates, "blasting_delay_hours": [float(i % 3) for i in range(30)], "planned_blasts": [1] * 30})

    def production(self):
        return self._production.copy()

    def weather(self):
        return self._weather.copy()

    def equipment(self):
        return self._equipment.copy()

    def blasting(self):
        return self._blasting.copy()


def test_summary_is_deterministic_and_one_row_per_date():
    service = OperationsSummaryService(store=FakeStore())
    result = service.summary(days=30)
    assert result["production_records"] == 30
    assert result["data_coverage"]["aligned_days"] == 30
    assert len(result["production_associations"]) == 3
    assert len(result["risk_signals"]) == 4
    assert service.summary(days=30) == result


def test_summary_window_does_not_use_future_rows():
    store = FakeStore()
    service = OperationsSummaryService(store=store)
    baseline = service.summary(days=7)
    store._production.loc[len(store._production)] = [pd.Timestamp("2026-02-01"), 9999.0, 1.0]
    store._weather.loc[len(store._weather)] = [pd.Timestamp("2026-02-01"), 9999.0, 0.99, 70.0]
    changed = service.summary(days=7)
    assert changed["latest_date"] == "2026-01-30"
    assert changed["production_records"] == baseline["production_records"]
    assert changed["production_mt_mean"] == baseline["production_mt_mean"]


def test_summary_handles_missing_operational_dates_without_row_explosion():
    store = FakeStore()
    store._weather = store._weather.iloc[:-5].copy()
    store._equipment = store._equipment.iloc[:-4].copy()
    store._blasting = store._blasting.iloc[:-3].copy()
    result = OperationsSummaryService(store=store).summary(days=30)
    assert result["data_coverage"]["aligned_days"] == 30
    assert result["data_coverage"]["weather"] == 25
    assert result["data_coverage"]["equipment"] == 26
    assert result["data_coverage"]["blasting"] == 27


def test_operations_summary_api():
    with TestClient(app) as client:
        response = client.get("/api/v1/operations/summary?days=30")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_mode"] == "demo"
    assert payload["synthetic_data"] is True
    assert payload["analysis_window_days"] == 30
    assert payload["overall_operational_risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(payload["production_associations"]) == 3
    assert len(payload["risk_signals"]) == 4


def test_operations_summary_rejects_invalid_window():
    with TestClient(app) as client:
        response = client.get("/api/v1/operations/summary?days=5")
    assert response.status_code == 422
