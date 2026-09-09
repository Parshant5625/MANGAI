from __future__ import annotations

from backend.app.services.recommendations import RecommendationService


class FakeProduction:
    def forecast(self, site_id=None, horizon=7):
        return {"shortfall_probability": 0.9, "gap_mt": -100.0}


class FakeOperations:
    def equipment(self, site_id=None):
        return {"items": [
            {"equipment_id": "E1", "status": "CRITICAL", "downtime_7d_hours": 20.0, "utilization": 0.4},
            {"equipment_id": "E2", "status": "NORMAL", "downtime_7d_hours": 2.0, "utilization": 0.5},
        ]}

    def weather(self, site_id=None):
        return {"weather_risk": "HIGH", "rainfall_7d_mm": 100.0, "soil_moisture": 0.6}

    def blasting(self, site_id=None):
        return {"planned_blasts_7d": 2, "delay_hours_7d": 10.0, "delay_trend": "WORSENING"}


class FakeReserve:
    def get_summary(self, site_id=None):
        return {"very_high_prospectivity_cells": 3, "average_probability": 0.8, "prototype_resource_potential": {"expected_tonnage": 1000.0}}


def service():
    return RecommendationService(FakeProduction(), FakeOperations(), FakeReserve())


def test_builds_evidence_driven_recommendations():
    result = service().list_recommendations()
    assert result["recommendation_count"] == 5
    assert result["recommendations"][0]["priority"] == "CRITICAL"
    assert all(item["status"] == "PROPOSED" for item in result["recommendations"])
    assert all(item["suggested_window"]["requires_human_approval"] is True for item in result["recommendations"])
    assert all("basis" in item["estimated_impact"] for item in result["recommendations"])


def test_equipment_impact_is_bounded_by_forecast_gap():
    result = service().list_recommendations()
    item = next(x for x in result["recommendations"] if x["category"] == "EQUIPMENT")
    low, high = item["estimated_impact"]["production_recovery_mt"]
    assert 0 <= low <= high <= 100


def test_simulation_does_not_mutate_baseline_and_records_scenario():
    svc = service()
    baseline = svc.list_recommendations()
    simulated = svc.simulate({"reduce_downtime_pct": 25, "defer_weather_sensitive_blasts": True})
    assert baseline["recommendations"][0]["status"] == "PROPOSED"
    assert simulated["recommendations"][0]["status"] == "SIMULATED"
    assert simulated["scenario"]["reduce_downtime_pct"] == 25.0
    assert simulated["scenario"]["defer_weather_sensitive_blasts"] is True


def test_simulation_clamps_invalid_downtime_percentage():
    result = service().simulate({"reduce_downtime_pct": 500})
    assert result["scenario"]["reduce_downtime_pct"] == 100.0
