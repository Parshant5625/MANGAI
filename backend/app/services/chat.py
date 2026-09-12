from __future__ import annotations

from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.operations import OperationsService
from backend.app.services.operations_summary import OperationsSummaryService
from backend.app.services.production import ProductionService
from backend.app.services.recommendations import RecommendationService
from backend.app.services.reserve import ReserveService


class ChatService:
    """Deterministic, evidence-first assistant over MANGAI's existing services."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _match(message: str, terms: tuple[str, ...]) -> bool:
        text = message.lower()
        return any(term in text for term in terms)

    @staticmethod
    def _pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:,.0f}"

    def answer(self, message: str, site_id: str | None = None) -> dict[str, Any]:
        text = message.strip()
        settings = self.settings
        common = {
            "data_mode": settings.data_mode,
            "synthetic_data": settings.data_mode == "demo",
            "disclaimer": "MANGAI provides decision support; operational actions require human approval. Prototype resource potential is not an official mineral-reserve classification.",
        }

        if self._match(text, ("hello", "hi", "hey", "help", "what can you do")):
            return {**common, "intent": "help", "confidence": 0.99, "answer": "I can explain MANGAI's reserve intelligence, production forecast, equipment health, weather and blasting risk, recommendations, and model health. Ask me a specific question or use one of the suggested prompts below.", "evidence": [], "suggested_questions": self.suggestions()}

        if self._match(text, ("production", "forecast", "shortfall", "target", "risk", "output")):
            forecast = ProductionService().forecast(site_id=site_id, horizon=7)
            risk = float(forecast["shortfall_probability"])
            drivers = forecast.get("top_drivers", [])[:3]
            driver_text = ", ".join(str(d.get("feature", "driver")).replace("_", " ") for d in drivers) or "no dominant drivers returned"
            return {**common, "intent": "production", "confidence": 0.96, "answer": f"The 7-day production forecast is {self._fmt(float(forecast['forecast_mt']))} t against a target of {self._fmt(float(forecast['target_mt']))} t. The modeled shortfall probability is {self._pct(risk)} ({forecast['severity']}). The strongest current drivers are {driver_text}.", "evidence": [{"label": "7-day forecast", "value": f"{self._fmt(float(forecast['forecast_mt']))} t"}, {"label": "7-day target", "value": f"{self._fmt(float(forecast['target_mt']))} t"}, {"label": "Gap", "value": f"{self._fmt(float(forecast['gap_mt']))} t"}, {"label": "Shortfall probability", "value": self._pct(risk)}, {"label": "Severity", "value": str(forecast["severity"])}], "suggested_questions": ["Why is production risk high?", "Which equipment is driving downtime?", "What actions are recommended?"]}

        if self._match(text, ("equipment", "fleet", "downtime", "excavator", "availability", "utilization", "machine")):
            equipment = OperationsService().equipment(site_id=site_id)
            items = sorted(equipment.get("items", []), key=lambda item: float(item.get("downtime_7d_hours", 0)), reverse=True)
            details = "; ".join(f"{item.get('equipment_id', 'asset')} ({float(item.get('downtime_7d_hours', 0)):.1f} h downtime)" for item in items[:3])
            return {**common, "intent": "equipment", "confidence": 0.95, "answer": f"Fleet availability is {self._pct(float(equipment['fleet_availability']))} and utilization is {self._pct(float(equipment['fleet_utilization']))}. {equipment['critical_equipment_count']} critical asset(s) are flagged. Highest recent downtime: {details or 'no asset ranking available'}.", "evidence": [{"label": "Fleet availability", "value": self._pct(float(equipment["fleet_availability"]))}, {"label": "Fleet utilization", "value": self._pct(float(equipment["fleet_utilization"]))}, {"label": "Critical equipment", "value": str(equipment["critical_equipment_count"])}], "suggested_questions": ["Why is production risk high?", "Which assets need attention first?", "What are the current recommendations?"]}

        if self._match(text, ("weather", "rain", "rainfall", "soil moisture", "blast", "blasting")):
            ops = OperationsService()
            weather = ops.weather(site_id=site_id)
            blasting = ops.blasting(site_id=site_id)
            return {**common, "intent": "weather_blasting", "confidence": 0.94, "answer": f"Recent weather shows {float(weather['rainfall_7d_mm']):.1f} mm of 7-day rainfall with soil moisture at {self._pct(float(weather['soil_moisture']))}. There are {blasting['planned_blasts_7d']} planned blasts in the current window and the blast overlap risk is {blasting['overlap_risk']}.", "evidence": [{"label": "7-day rainfall", "value": f"{float(weather['rainfall_7d_mm']):.1f} mm"}, {"label": "Soil moisture", "value": self._pct(float(weather["soil_moisture"]))}, {"label": "Planned blasts", "value": str(blasting["planned_blasts_7d"])}, {"label": "Overlap risk", "value": str(blasting["overlap_risk"])}], "suggested_questions": ["Are any blasts weather-sensitive?", "What is the production forecast?", "Show me recommended actions."]}

        if self._match(text, ("reserve", "prospectivity", "ore", "manganese", "mn", "grade", "thickness", "resource", "target area")):
            summary = ReserveService().get_summary(site_id=site_id)
            cells = ReserveService().get_prospectivity(site_id=site_id, min_probability=0.55, limit=50).get("cells", [])
            top = max(cells, key=lambda cell: float(cell.get("probability", 0)), default=None)
            target = f" Top target {top.get('id')} has {self._pct(float(top.get('probability', 0)))} prospectivity." if top else ""
            return {**common, "intent": "reserve", "confidence": 0.92, "answer": f"MANGAI currently identifies {summary['high_prospectivity_cells']} high-prospectivity and {summary['very_high_prospectivity_cells']} very-high cells. Average predicted grade is {float(summary['average_predicted_grade_pct']):.1f}% Mn and average predicted thickness is {float(summary['average_predicted_thickness_m']):.1f} m.{target}", "evidence": [{"label": "High prospectivity cells", "value": str(summary["high_prospectivity_cells"])}, {"label": "Very-high cells", "value": str(summary["very_high_prospectivity_cells"])}, {"label": "Average predicted grade", "value": f"{float(summary['average_predicted_grade_pct']):.1f}% Mn"}, {"label": "Average predicted thickness", "value": f"{float(summary['average_predicted_thickness_m']):.1f} m"}], "suggested_questions": ["Explain the highest prospectivity target.", "How confident is the reserve prediction?", "What data supports this target?"]}

        if self._match(text, ("recommendation", "recommend", "action", "what should", "what do we do", "next action")):
            recommendations = RecommendationService().list_recommendations(site_id=site_id)
            items = recommendations.get("recommendations", [])
            top = sorted(items, key=lambda item: float(item.get("confidence", 0)), reverse=True)[:3]
            titles = "; ".join(str(item.get("title", "Action")) for item in top)
            return {**common, "intent": "recommendations", "confidence": 0.93, "answer": f"MANGAI has {len(items)} recommendation(s) available. Highest-confidence actions include: {titles or 'no active recommendations returned'}. These are decision-support suggestions and require human approval before execution.", "evidence": [{"label": "Active recommendations", "value": str(len(items))}] + [{"label": "Action", "value": str(item.get("title", "Recommendation"))} for item in top], "suggested_questions": ["Why is production risk high?", "Which equipment needs attention first?", "What is the reserve priority area?"]}

        if self._match(text, ("health", "model health", "drift", "data quality", "model status")):
            operations = OperationsSummaryService().summary(site_id=site_id, days=30)
            return {**common, "intent": "health", "confidence": 0.9, "answer": f"MANGAI is running in {settings.data_mode.upper()} mode. The 30-day operations summary contains {operations.get('records', 0)} records. For detailed model drift and data-quality diagnostics, open the Health page.", "evidence": [{"label": "Data mode", "value": settings.data_mode.upper()}, {"label": "Operations records", "value": str(operations.get("records", 0))}], "suggested_questions": ["What is the production risk?", "What reserve targets are strongest?", "What can you do?"]}

        return {**common, "intent": "unknown", "confidence": 0.55, "answer": "I can answer questions about reserve prospectivity, production forecasts and shortfall risk, equipment, weather/blasting, recommendations, and model/data health. Try ‘Why is production risk high?’ or ‘Which reserve area should we investigate first?’", "evidence": [], "suggested_questions": self.suggestions()}

    @staticmethod
    def suggestions() -> list[str]:
        return ["Why is production risk high?", "Which equipment has the highest downtime?", "What is the 7-day production forecast?", "Which reserve area should we investigate first?", "What are the current recommended actions?", "What is the weather and blasting risk?"]
