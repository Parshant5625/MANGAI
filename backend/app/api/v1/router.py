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
from backend.app.adapters.satellite.local_file import LocalFileSatelliteProvider
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
def get_reserve_prospectivity(site_id: str | None = None, bbox: str | None = None, min_probability: Annotated[float | None, Query(ge=0, le=1)] = None, limit: Annotated[int, Query(ge=1, le=2000)] = 500) -> dict:
    try:
        return ReserveService().get_prospectivity(site_id=site_id, bbox=bbox, min_probability=min_probability, limit=limit)
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
def get_production_forecast(site_id: str | None = None, horizon: Annotated[int, Query(ge=1, le=30)] = 7) -> dict:
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

@router.get("/satellite/scenes", tags=["satellite"])
def get_satellite_scenes(site_id: str | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> dict:
    settings = get_settings()
    observations = LocalFileSatelliteProvider().fetch_observations(site_id or settings.demo_site_id, "", "")
    scenes = []
    for index, observation in enumerate(observations[:limit]):
        source = str(observation.get("source", observation.get("provider", "Satellite")))
        scenes.append({
            "id": str(observation.get("scene_id", observation.get("id", f"demo-scene-{index}"))),
            "source": source,
            "date": str(observation.get("date", observation.get("timestamp", ""))),
            "cloud_cover": float(observation.get("cloud_cover", 0) or 0),
            "quality": float(observation.get("quality", 0.8) or 0.8),
            "bands": [str(k) for k in observation.keys() if str(k).lower().startswith(("b", "nir", "red", "swir"))][:12],
            "thermal_coverage": float(observation.get("thermal_coverage", 0) or 0) if "thermal_coverage" in observation else None,
        })
    return {"site_id": site_id or settings.demo_site_id, "count": len(scenes), "scenes": scenes, "data_mode": "DEMO / LOCAL FILE"}

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

@router.post("/chat", tags=["copilot"])
def chat(payload: dict) -> dict:
    question = str(payload.get("message", "")).strip()
    context = payload.get("context") or {}
    risk = float(context.get("production_risk", 0) or 0)
    gap = float(context.get("production_gap_mt", 0) or 0)
    fleet = float(context.get("fleet_utilization", 0) or 0)
    quality = float(context.get("data_quality", 0) or 0)
    target = context.get("selected_target")
    q = question.lower()
    evidence = [f"shortfall risk is currently {risk:.0%}", f"production gap is {gap:,.0f} t", f"fleet utilization is {fleet:.0%}", f"data quality is {quality:.0%}"]
    if target:
        evidence.append(f"selected target is {target}")
    if "why" in q and ("production" in q or "risk" in q):
        answer = "The current production-risk signal is driven by the operating evidence exposed to the dashboard: " + "; ".join(evidence[:4]) + ". Investigate the highest-ranked production driver, then cross-check equipment downtime and blasting/weather constraints before changing the mine plan."
    elif "next" in q or "investigate" in q:
        answer = "Recommended investigation order: 1) validate the top production driver, 2) inspect critical-equipment downtime, 3) check weather/blasting overlap, 4) compare the proposed action with the current production target. Do not treat the model output as a clearance or mine-plan approval."
    elif "evidence" in q or "action" in q:
        answer = "The strongest dashboard evidence currently available is: " + "; ".join(evidence) + ". Use these signals to prioritize investigation; the final operational decision remains with the accountable mine team."
    else:
        answer = "I can explain the current production risk, identify the next investigation step, or summarize evidence for a recommendation. Current evidence: " + "; ".join(evidence) + "."
    confidence = max(0.45, min(0.95, 0.55 + quality * 0.25))
    return {"answer": answer, "confidence": confidence, "evidence": evidence, "mode": "bounded-decision-support"}
