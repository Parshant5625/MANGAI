from __future__ import annotations

from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.demo_data import demo_envelope
from backend.app.services.operations import OperationsService
from backend.app.services.production import ProductionService
from backend.app.services.reserve import ReserveService

PRIORITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


class RecommendationService:
    def __init__(
        self,
        production_service: ProductionService | None = None,
        operations_service: OperationsService | None = None,
        reserve_service: ReserveService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.production_service = production_service or ProductionService()
        self.operations_service = operations_service or OperationsService()
        self.reserve_service = reserve_service or ReserveService()

    def list_recommendations(self, site_id: str | None = None) -> dict[str, Any]:
        forecast = self.production_service.forecast(site_id=site_id, horizon=7)
        equipment = self.operations_service.equipment(site_id=site_id)
        weather = self.operations_service.weather(site_id=site_id)
        blasting = self.operations_service.blasting(site_id=site_id)
        reserve = self.reserve_service.get_summary(site_id=site_id)
        recommendations = self._build(forecast, equipment, weather, blasting, reserve)
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "recommendations": recommendations,
        }

    def simulate(self, payload: dict[str, Any], site_id: str | None = None) -> dict[str, Any]:
        response = self.list_recommendations(site_id=site_id)
        adjusted = []
        reduce_downtime_pct = float(payload.get("reduce_downtime_pct", 0))
        rainfall_override = payload.get("rainfall_risk_override")
        for item in response["recommendations"]:
            item = dict(item)
            item["status"] = "SIMULATED"
            impact = dict(item["estimated_impact"])
            if reduce_downtime_pct and item["category"] == "EQUIPMENT":
                current_range = impact.get("production_recovery_mt", [0, 0])
                lift = 1 + reduce_downtime_pct / 100
                impact["production_recovery_mt"] = [round(current_range[0] * lift, 2), round(current_range[1] * lift, 2)]
                item["confidence"] = min(0.95, round(float(item["confidence"]) + 0.04, 2))
            if payload.get("defer_weather_sensitive_blasts") and item["category"] in {"WEATHER", "BLASTING"}:
                impact["delay_risk_reduction_pct"] = [12, 24]
                item["priority"] = "MEDIUM" if item["priority"] == "HIGH" else item["priority"]
            if rainfall_override == "LOW" and item["category"] == "WEATHER":
                item["priority"] = "LOW"
                item["confidence"] = round(max(0.4, float(item["confidence"]) - 0.1), 2)
            item["estimated_impact"] = impact
            adjusted.append(item)
        response["recommendations"] = sorted(
            adjusted, key=lambda item: (PRIORITY_ORDER[item["priority"]], item["confidence"]), reverse=True
        )
        return response

    def _build(
        self,
        forecast: dict[str, Any],
        equipment: dict[str, Any],
        weather: dict[str, Any],
        blasting: dict[str, Any],
        reserve: dict[str, Any],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        shortfall_probability = float(forecast["shortfall_probability"])
        critical = [item for item in equipment["items"] if item["status"] == "CRITICAL"]
        if shortfall_probability >= 0.55 and critical:
            top = critical[0]
            spare = [item for item in equipment["items"] if item["status"] == "NORMAL" and item["utilization"] < 0.7]
            recovery_low = round(max(80.0, top["downtime_7d_hours"] * 8), 1)
            recovery_high = round(max(160.0, top["downtime_7d_hours"] * 18), 1)
            items.append(
                {
                    "id": "REC-DEMO-0001",
                    "category": "EQUIPMENT",
                    "priority": "HIGH" if shortfall_probability < 0.8 else "CRITICAL",
                    "title": f"Prioritize {top['equipment_id']} maintenance and capacity review",
                    "rationale": "Recent downtime is elevated while the production forecast is below target. This is a planning prompt, not an operational dispatch instruction.",
                    "evidence": {
                        "equipment_id": top["equipment_id"],
                        "downtime_7d_hours": top["downtime_7d_hours"],
                        "utilization": top["utilization"],
                        "shortfall_probability": shortfall_probability,
                        "forecast_gap_mt": forecast["gap_mt"],
                        "spare_capacity_assets": len(spare),
                    },
                    "estimated_impact": {"production_recovery_mt": [recovery_low, recovery_high]},
                    "confidence": round(min(0.9, 0.58 + shortfall_probability * 0.3), 2),
                    "affected_equipment": [top["equipment_id"]],
                    "affected_area": None,
                    "suggested_window": {"label": "next_24h", "requires_human_approval": True},
                    "status": "PROPOSED",
                }
            )
            if spare:
                items.append(
                    {
                        "id": "REC-DEMO-0005",
                        "category": "PRODUCTION",
                        "priority": "MEDIUM",
                        "title": f"Review redeploying available capacity from {spare[0]['equipment_id']}",
                        "rationale": "A lower-utilization asset may offset constrained capacity after human review of site conditions.",
                        "evidence": {
                            "available_asset": spare[0]["equipment_id"],
                            "available_utilization": spare[0]["utilization"],
                            "constrained_asset": top["equipment_id"],
                            "shortfall_probability": shortfall_probability,
                        },
                        "estimated_impact": {"production_recovery_mt": [round(recovery_low * 0.4, 1), round(recovery_high * 0.6, 1)]},
                        "confidence": 0.61,
                        "affected_equipment": [spare[0]["equipment_id"], top["equipment_id"]],
                        "affected_area": "active fleet",
                        "suggested_window": {"label": "next_shift_planning", "requires_human_approval": True},
                        "status": "PROPOSED",
                    }
                )
        if weather["weather_risk"] in {"MEDIUM", "HIGH"} and blasting["planned_blasts_7d"] > 0:
            items.append(
                {
                    "id": "REC-DEMO-0002",
                    "category": "WEATHER",
                    "priority": "HIGH" if weather["weather_risk"] == "HIGH" else "MEDIUM",
                    "title": "Review weather-sensitive blast windows",
                    "rationale": "Rainfall and soil-moisture risk overlaps with planned blasting activity.",
                    "evidence": {
                        "weather_risk": weather["weather_risk"],
                        "rainfall_7d_mm": weather["rainfall_7d_mm"],
                        "soil_moisture": weather["soil_moisture"],
                        "planned_blasts_7d": blasting["planned_blasts_7d"],
                    },
                    "estimated_impact": {"delay_risk_reduction_pct": [8, 18]},
                    "confidence": 0.72 if weather["weather_risk"] == "HIGH" else 0.62,
                    "affected_equipment": [],
                    "affected_area": "active pit schedule",
                    "suggested_window": {"label": "before_next_planned_blast", "requires_human_approval": True},
                    "status": "PROPOSED",
                }
            )
        if blasting["delay_trend"] == "WORSENING" and forecast["gap_mt"] < 0:
            items.append(
                {
                    "id": "REC-DEMO-0003",
                    "category": "BLASTING",
                    "priority": "MEDIUM",
                    "title": "Review blast preparation delays",
                    "rationale": "Blast delays are rising during a period with a negative production gap.",
                    "evidence": {
                        "delay_hours_7d": blasting["delay_hours_7d"],
                        "delay_trend": blasting["delay_trend"],
                        "forecast_gap_mt": forecast["gap_mt"],
                    },
                    "estimated_impact": {"schedule_recovery_hours": [4, 10]},
                    "confidence": 0.65,
                    "affected_equipment": [],
                    "affected_area": "blast planning",
                    "suggested_window": {"label": "next_shift_planning", "requires_human_approval": True},
                    "status": "PROPOSED",
                }
            )
        if reserve["very_high_prospectivity_cells"] > 0:
            items.append(
                {
                    "id": "REC-DEMO-0004",
                    "category": "RESERVE",
                    "priority": "MEDIUM",
                    "title": "Prioritize investigation of very high prospectivity cells",
                    "rationale": "Synthetic geospatial model highlights zones with high probability, grade proxy, and thickness proxy.",
                    "evidence": {
                        "very_high_prospectivity_cells": reserve["very_high_prospectivity_cells"],
                        "average_probability": reserve["average_probability"],
                        "prototype_resource_potential_t": reserve["prototype_resource_potential"]["expected_tonnage"],
                    },
                    "estimated_impact": {"investigation_targets": reserve["very_high_prospectivity_cells"]},
                    "confidence": 0.68,
                    "affected_equipment": [],
                    "affected_area": "reserve grid",
                    "suggested_window": {"label": "next_geology_review", "requires_human_approval": True},
                    "status": "PROPOSED",
                }
            )
        items = sorted(items, key=lambda item: (PRIORITY_ORDER[item["priority"]], item["confidence"]), reverse=True)
        return items

