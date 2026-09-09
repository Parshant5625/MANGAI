from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.demo_data import demo_envelope
from backend.app.services.operations import OperationsService
from backend.app.services.production import ProductionService
from backend.app.services.reserve import ReserveService

PRIORITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


class RecommendationService:
    """Evidence-driven planning recommendations.

    Recommendations are decision-support prompts, not autonomous dispatch or
    mine-control instructions. Impact estimates are scenario estimates and
    must be validated by an authorized operational/geology team.
    """

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

    @staticmethod
    def _confidence(*values: float, minimum: float = 0.4, maximum: float = 0.95) -> float:
        valid = [max(0.0, min(1.0, float(v))) for v in values]
        return round(min(maximum, max(minimum, sum(valid) / len(valid))), 2) if valid else minimum

    @staticmethod
    def _approval_window(label: str) -> dict[str, Any]:
        return {"label": label, "requires_human_approval": True}

    @staticmethod
    def _impact_range(base: float, low_factor: float, high_factor: float) -> list[float]:
        base = max(0.0, float(base))
        return [round(base * low_factor, 2), round(base * high_factor, 2)]

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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
            "methodology_note": "Recommendations are generated from forecast, operational, weather, blasting and reserve evidence. Impact ranges are scenario estimates, not guarantees. Human approval is required for every proposed action.",
        }

    def simulate(self, payload: dict[str, Any], site_id: str | None = None) -> dict[str, Any]:
        baseline = self.list_recommendations(site_id=site_id)
        reduce_downtime_pct = max(0.0, min(100.0, float(payload.get("reduce_downtime_pct", 0) or 0)))
        defer_blasts = bool(payload.get("defer_weather_sensitive_blasts", False))
        rainfall_override = payload.get("rainfall_risk_override")

        adjusted: list[dict[str, Any]] = []
        for original in baseline["recommendations"]:
            item = dict(original)
            item["evidence"] = dict(original["evidence"])
            impact = dict(original["estimated_impact"])
            item["status"] = "SIMULATED"

            if reduce_downtime_pct and item["category"] == "EQUIPMENT":
                current = impact.get("production_recovery_mt", [0.0, 0.0])
                multiplier = 1.0 + reduce_downtime_pct / 100.0
                impact["production_recovery_mt"] = [round(float(current[0]) * multiplier, 2), round(float(current[1]) * multiplier, 2)]
                item["evidence"]["scenario_reduce_downtime_pct"] = reduce_downtime_pct

            if defer_blasts and item["category"] in {"WEATHER", "BLASTING"}:
                impact["delay_risk_reduction_pct"] = [12.0, 24.0]
                item["evidence"]["scenario_defer_weather_sensitive_blasts"] = True

            if rainfall_override == "LOW" and item["category"] == "WEATHER":
                item["priority"] = "LOW"
                item["confidence"] = round(max(0.4, float(item["confidence"]) - 0.1), 2)
                item["evidence"]["scenario_rainfall_risk_override"] = "LOW"

            item["estimated_impact"] = impact
            adjusted.append(item)

        adjusted.sort(key=lambda item: (PRIORITY_ORDER[item["priority"]], item["confidence"]), reverse=True)
        baseline["scenario"] = {
            "reduce_downtime_pct": reduce_downtime_pct,
            "defer_weather_sensitive_blasts": defer_blasts,
            "rainfall_risk_override": rainfall_override,
        }
        baseline["recommendations"] = adjusted
        baseline["recommendation_count"] = len(adjusted)
        return baseline

    def _build(
        self,
        forecast: dict[str, Any],
        equipment: dict[str, Any],
        weather: dict[str, Any],
        blasting: dict[str, Any],
        reserve: dict[str, Any],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        shortfall_probability = float(forecast.get("shortfall_probability", 0.0))
        forecast_gap = float(forecast.get("gap_mt", 0.0))
        critical = [item for item in equipment.get("items", []) if item.get("status") == "CRITICAL"]
        if shortfall_probability >= 0.55 and critical:
            top = critical[0]
            spare = [item for item in equipment.get("items", []) if item.get("status") == "NORMAL" and float(item.get("utilization", 1.0)) < 0.7]
            downtime = max(0.0, float(top.get("downtime_7d_hours", 0.0)))
            recovery_base = min(abs(forecast_gap), downtime * 8.0) if forecast_gap < 0 else downtime * 8.0
            recovery = self._impact_range(recovery_base, 0.35, 0.75)
            items.append({
                "id": "REC-DEMO-0001", "category": "EQUIPMENT",
                "priority": "HIGH" if shortfall_probability < 0.8 else "CRITICAL",
                "title": f"Prioritize {top['equipment_id']} maintenance and capacity review",
                "rationale": "Recent equipment downtime overlaps with a below-target production forecast.",
                "evidence": {"equipment_id": top["equipment_id"], "downtime_7d_hours": downtime, "utilization": top.get("utilization"), "shortfall_probability": shortfall_probability, "forecast_gap_mt": forecast_gap, "spare_capacity_assets": len(spare)},
                "estimated_impact": {"production_recovery_mt": recovery, "basis": "bounded downtime-to-production scenario estimate"},
                "confidence": self._confidence(shortfall_probability, 1.0 if downtime >= 12 else downtime / 12),
                "affected_equipment": [top["equipment_id"]], "affected_area": None,
                "suggested_window": self._approval_window("next_24h"), "status": "PROPOSED",
            })
            if spare:
                items.append({
                    "id": "REC-DEMO-0005", "category": "PRODUCTION", "priority": "MEDIUM",
                    "title": f"Review redeploying available capacity from {spare[0]['equipment_id']}",
                    "rationale": "A lower-utilization asset may offset constrained capacity after site and safety review.",
                    "evidence": {"available_asset": spare[0]["equipment_id"], "available_utilization": spare[0]["utilization"], "constrained_asset": top["equipment_id"], "shortfall_probability": shortfall_probability},
                    "estimated_impact": {"production_recovery_mt": self._impact_range(recovery[1], 0.4, 0.7), "basis": "bounded capacity-redeployment scenario estimate"},
                    "confidence": 0.61,
                    "affected_equipment": [spare[0]["equipment_id"], top["equipment_id"]], "affected_area": "active fleet",
                    "suggested_window": self._approval_window("next_shift_planning"), "status": "PROPOSED",
                })

        if weather.get("weather_risk") in {"MEDIUM", "HIGH"} and int(blasting.get("planned_blasts_7d", 0)) > 0:
            risk = weather["weather_risk"]
            items.append({
                "id": "REC-DEMO-0002", "category": "WEATHER", "priority": "HIGH" if risk == "HIGH" else "MEDIUM",
                "title": "Review weather-sensitive blast windows",
                "rationale": "Recent rainfall/soil-moisture exposure overlaps with planned blasting activity.",
                "evidence": {"weather_risk": risk, "rainfall_7d_mm": weather.get("rainfall_7d_mm"), "soil_moisture": weather.get("soil_moisture"), "planned_blasts_7d": blasting.get("planned_blasts_7d")},
                "estimated_impact": {"delay_risk_reduction_pct": [8.0, 18.0], "basis": "scenario estimate from weather-sensitive scheduling"},
                "confidence": 0.72 if risk == "HIGH" else 0.62,
                "affected_equipment": [], "affected_area": "active pit schedule",
                "suggested_window": self._approval_window("before_next_planned_blast"), "status": "PROPOSED",
            })

        if blasting.get("delay_trend") == "WORSENING" and forecast_gap < 0:
            delay = max(0.0, float(blasting.get("delay_hours_7d", 0.0)))
            items.append({
                "id": "REC-DEMO-0003", "category": "BLASTING", "priority": "MEDIUM",
                "title": "Review blast preparation delays",
                "rationale": "Blast delays are worsening while the production forecast is below target.",
                "evidence": {"delay_hours_7d": delay, "delay_trend": blasting["delay_trend"], "forecast_gap_mt": forecast_gap},
                "estimated_impact": {"schedule_recovery_hours": self._impact_range(delay, 0.25, 0.65), "basis": "bounded delay-reduction scenario estimate"},
                "confidence": 0.65,
                "affected_equipment": [], "affected_area": "blast planning",
                "suggested_window": self._approval_window("next_shift_planning"), "status": "PROPOSED",
            })

        very_high = int(reserve.get("very_high_prospectivity_cells", 0))
        if very_high > 0:
            items.append({
                "id": "REC-DEMO-0004", "category": "RESERVE", "priority": "MEDIUM",
                "title": "Prioritize investigation of very high prospectivity cells",
                "rationale": "The geospatial model identifies high-prospectivity zones suitable for investigation prioritization.",
                "evidence": {"very_high_prospectivity_cells": very_high, "average_probability": reserve.get("average_probability"), "prototype_resource_potential_t": reserve.get("prototype_resource_potential", {}).get("expected_tonnage")},
                "estimated_impact": {"investigation_targets": very_high, "basis": "target-prioritization scenario; not a reserve declaration"},
                "confidence": 0.68,
                "affected_equipment": [], "affected_area": "reserve grid",
                "suggested_window": self._approval_window("next_geology_review"), "status": "PROPOSED",
            })

        return sorted(items, key=lambda item: (PRIORITY_ORDER[item["priority"]], item["confidence"]), reverse=True)
