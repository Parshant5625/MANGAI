from __future__ import annotations

from typing import Any

from backend.app.core.config import get_settings
from backend.app.services.data_quality import DataQualityService
from backend.app.services.demo_data import demo_envelope
from backend.app.services.model_registry import ModelRegistryService
from backend.app.services.operations import OperationsService
from backend.app.services.production import ProductionService
from backend.app.services.recommendations import RecommendationService
from backend.app.services.reserve import ReserveService


class OverviewService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.reserve_service = ReserveService()
        self.production_service = ProductionService()
        self.operations_service = OperationsService()
        self.data_quality_service = DataQualityService()
        self.model_registry_service = ModelRegistryService()
        self.recommendation_service = RecommendationService(
            production_service=self.production_service,
            operations_service=self.operations_service,
            reserve_service=self.reserve_service,
        )

    def get_overview(self, site_id: str | None = None) -> dict[str, Any]:
        site_id = site_id or self.settings.demo_site_id
        reserve = self.reserve_service.get_summary(site_id=site_id)
        production = self.production_service.forecast(site_id=site_id, horizon=7)
        equipment = self.operations_service.equipment(site_id=site_id)
        quality = self.data_quality_service.run()
        recommendations = self.recommendation_service.list_recommendations(site_id=site_id)
        models = self.model_registry_service.list_models()
        model_health = "READY" if models["models"] and quality["overall_score"] >= 0.8 else "WATCH"
        return {
            **demo_envelope(),
            "site_id": site_id,
            "site_name": self.settings.demo_site_name,
            "resource_potential_tonnage": reserve["prototype_resource_potential"]["expected_tonnage"],
            "high_prospectivity_area_ha": reserve["high_prospectivity_area_ha"],
            "next_7_day_production_mt": production["forecast_mt"],
            "shortfall_probability": production["shortfall_probability"],
            "production_gap_mt": production["gap_mt"],
            "critical_equipment_count": equipment["critical_equipment_count"],
            "recommendation_count": len(recommendations["recommendations"]),
            "model_health": model_health,
            "data_quality_score": quality["overall_score"],
            "kpis": {
                "cells_evaluated": reserve["cells_evaluated"],
                "very_high_prospectivity_cells": reserve["very_high_prospectivity_cells"],
                "fleet_availability": equipment["fleet_availability"],
                "fleet_utilization": equipment["fleet_utilization"],
                "forecast_origin": production["forecast_origin"],
            },
        }

