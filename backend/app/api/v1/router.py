from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from backend.app.core.config import get_settings
from backend.app.schemas import (
    BlastingResponse,
    DataQualityResponse,
    EquipmentResponse,
    ModelRegistryResponse,
    OverviewResponse,
    ProductionForecastResponse,
    ProductionPredictionRequest,
    RecommendationResponse,
    RecommendationSimulationRequest,
    ReservePredictionRequest,
    ReservePredictionResponse,
    ReserveProspectivityResponse,
    ReserveSummaryResponse,
    WeatherResponse,
)
from backend.app.services.data_quality import DataQualityService
from backend.app.services.model_registry import ModelRegistryService
from backend.app.services.operations import OperationsService
from backend.app.services.overview import OverviewService
from backend.app.services.production import ProductionService
from backend.app.services.recommendations import RecommendationService
from backend.app.services.reserve import ReserveService


router = APIRouter()


@router.get("/overview", response_model=OverviewResponse, tags=["overview"])
def get_overview(site_id: str | None = None) -> dict:
    return OverviewService().get_overview(site_id=site_id)


@router.get("/reserves/prospectivity", response_model=ReserveProspectivityResponse, tags=["reserve"])
def get_reserve_prospectivity(
    site_id: str | None = None,
    bbox: str | None = None,
    min_probability: Annotated[float | None, Query(ge=0, le=1)] = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> dict:
    try:
        return ReserveService().get_prospectivity(
            site_id=site_id,
            bbox=bbox,
            min_probability=min_probability,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/reserves/summary", response_model=ReserveSummaryResponse, tags=["reserve"])
def get_reserve_summary(site_id: str | None = None) -> dict:
    return ReserveService().get_summary(site_id=site_id)


@router.get("/reserves/boreholes", tags=["reserve"])
def get_reserve_boreholes(site_id: str | None = None, limit: Annotated[int, Query(ge=1, le=2000)] = 400) -> dict:
    return ReserveService().boreholes(site_id=site_id, limit=limit)


@router.get("/reserves/{reserve_id}", tags=["reserve"])
def get_reserve_detail(reserve_id: str, site_id: str | None = None) -> dict:
    try:
        return ReserveService().get_detail(reserve_id, site_id=site_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Reserve cell not found: {reserve_id}") from exc


@router.get("/production/forecast", response_model=ProductionForecastResponse, tags=["production"])
def get_production_forecast(
    site_id: str | None = None,
    horizon: Annotated[int, Query(ge=1, le=30)] = 7,
) -> dict:
    return ProductionService().forecast(site_id=site_id, horizon=horizon)


@router.get("/production/risk", response_model=ProductionForecastResponse, tags=["production"])
def get_production_risk(site_id: str | None = None) -> dict:
    return ProductionService().risk(site_id=site_id)


@router.get("/production/history", tags=["production"])
def get_production_history(days: Annotated[int, Query(ge=7, le=365)] = 60) -> dict:
    settings = get_settings()
    return {"site_id": settings.demo_site_id, "records": ProductionService().history(days=days)}


@router.get("/equipment", response_model=EquipmentResponse, tags=["operations"])
def get_equipment(site_id: str | None = None) -> dict:
    return OperationsService().equipment(site_id=site_id)


@router.get("/weather", response_model=WeatherResponse, tags=["operations"])
def get_weather(site_id: str | None = None) -> dict:
    return OperationsService().weather(site_id=site_id)


@router.get("/blasting", response_model=BlastingResponse, tags=["operations"])
def get_blasting(site_id: str | None = None) -> dict:
    return OperationsService().blasting(site_id=site_id)


@router.get("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
def get_recommendations(site_id: str | None = None) -> dict:
    return RecommendationService().list_recommendations(site_id=site_id)


@router.get("/models", response_model=ModelRegistryResponse, tags=["mlops"])
def get_models() -> dict:
    return ModelRegistryService().list_models()


@router.get("/data-quality", response_model=DataQualityResponse, tags=["mlops"])
def get_data_quality() -> dict:
    return DataQualityService().run()


@router.post("/predictions/reserve", response_model=ReservePredictionResponse, tags=["prediction"])
def predict_reserve(payload: ReservePredictionRequest) -> dict:
    return ReserveService().predict(payload.model_dump())


@router.post("/predictions/production", response_model=ProductionForecastResponse, tags=["prediction"])
def predict_production(payload: ProductionPredictionRequest) -> dict:
    return ProductionService().predict(payload.model_dump())


@router.post("/recommendations/simulate", response_model=RecommendationResponse, tags=["recommendations"])
def simulate_recommendations(payload: RecommendationSimulationRequest, site_id: str | None = None) -> dict:
    return RecommendationService().simulate(payload.model_dump(), site_id=site_id)

