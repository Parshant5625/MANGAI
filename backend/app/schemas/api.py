from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DemoEnvelope(BaseModel):
    data_mode: str = "demo"
    synthetic_data: bool = True
    boundary_notice: str


class TopDriver(BaseModel):
    feature: str
    direction: Literal["positive", "negative", "neutral"]
    importance: float
    value: float | str | None = None


class PredictionInterval(BaseModel):
    p10: float
    p50: float
    p90: float


class ResourcePotential(BaseModel):
    label: str = "prototype resource potential"
    expected_tonnage: float
    p10: float
    p50: float
    p90: float
    assumptions: dict[str, Any]


class ProspectivityCell(BaseModel):
    id: str
    latitude: float
    longitude: float
    probability: float
    prospectivity_class: str
    predicted_grade_pct: float
    predicted_thickness_m: float
    confidence: float
    resource_potential: ResourcePotential
    top_contributors: list[TopDriver]
    data_support: dict[str, Any] = {}


class ReserveProspectivityResponse(DemoEnvelope):
    site_id: str
    bbox: list[float] | None = None
    count: int
    cells: list[ProspectivityCell]


class ReserveSummaryResponse(DemoEnvelope):
    site_id: str
    cells_evaluated: int
    high_prospectivity_cells: int
    very_high_prospectivity_cells: int
    high_prospectivity_area_ha: float
    average_probability: float
    average_predicted_grade_pct: float
    average_predicted_thickness_m: float
    prototype_resource_potential: ResourcePotential
    validation_note: str


class ReservePredictionRequest(BaseModel):
    latitude: float
    longitude: float
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    depth_m: float
    formation: str
    blue_b2: float = 0.12
    green_b3: float = 0.15
    red_b4: float = 0.16
    nir_b8: float = 0.35
    swir_b11: float = 0.28
    swir_b12: float = 0.22


class ReservePredictionResponse(DemoEnvelope):
    prediction: ProspectivityCell
    model_version: str


class ProductionForecastResponse(DemoEnvelope):
    site_id: str
    forecast_date: str
    forecast_origin: str
    horizon_days: int
    forecast_mt: float
    target_mt: float
    gap_mt: float
    shortfall_probability: float
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    prediction_interval: PredictionInterval
    top_drivers: list[TopDriver]
    model_version: str
    baseline_forecast_mt: float
    data_freshness: dict[str, Any]
    horizon_series: list[dict[str, Any]] = []


class ProductionPredictionRequest(BaseModel):
    horizon_days: int = Field(default=7, ge=1, le=30)
    rainfall_mm_7d: float = 0
    downtime_hours_7d: float = 0
    blasting_delay_7d: float = 0
    target_mt: float = 8500


class EquipmentItem(BaseModel):
    equipment_id: str
    equipment_type: str
    availability: float
    utilization: float
    downtime_7d_hours: float
    downtime_30d_hours: float
    maintenance_events_30d: int
    status: Literal["NORMAL", "WATCH", "CRITICAL"]


class EquipmentResponse(DemoEnvelope):
    site_id: str
    fleet_availability: float
    fleet_utilization: float
    critical_equipment_count: int
    items: list[EquipmentItem]
    maintenance_trend: list[dict[str, Any]] = []


class WeatherResponse(DemoEnvelope):
    site_id: str
    latest_date: str
    rainfall_7d_mm: float
    rainfall_30d_mm: float
    soil_moisture: float
    temperature_c: float
    weather_risk: Literal["LOW", "MEDIUM", "HIGH"]
    observations: list[dict[str, Any]]


class BlastingResponse(DemoEnvelope):
    site_id: str
    latest_date: str
    planned_blasts_7d: int
    delay_hours_7d: float
    delay_trend: Literal["IMPROVING", "STABLE", "WORSENING"]
    overlap_risk: Literal["LOW", "MEDIUM", "HIGH"]
    events: list[dict[str, Any]]


class Recommendation(BaseModel):
    id: str
    category: Literal["EQUIPMENT", "PRODUCTION", "WEATHER", "RESERVE", "BLASTING"]
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    title: str
    rationale: str
    evidence: dict[str, Any]
    estimated_impact: dict[str, Any]
    confidence: float
    affected_equipment: list[str] = []
    affected_area: str | None = None
    suggested_window: dict[str, Any]
    status: Literal["PROPOSED", "SIMULATED"] = "PROPOSED"


class RecommendationResponse(DemoEnvelope):
    site_id: str
    recommendations: list[Recommendation]


class RecommendationSimulationRequest(BaseModel):
    reduce_downtime_pct: float = Field(default=0, ge=0, le=75)
    rainfall_risk_override: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    defer_weather_sensitive_blasts: bool = False


class ModelVersion(BaseModel):
    model_name: str
    version: str
    task: str
    algorithm: str
    training_data_hash: str
    feature_schema_hash: str
    metrics: dict[str, Any]
    artifact_path: str
    created_at: str
    status: str
    notes: str = ""
    drift: dict[str, Any] = {}


class ModelRegistryResponse(DemoEnvelope):
    models: list[ModelVersion]


class DataQualityRun(BaseModel):
    dataset_name: str
    source: str
    row_count: int
    missing_rate: float
    duplicate_rate: float
    schema_valid: bool
    quality_score: float
    details: dict[str, Any]
    created_at: str


class DataQualityResponse(DemoEnvelope):
    overall_score: float
    runs: list[DataQualityRun]


class OverviewResponse(DemoEnvelope):
    site_id: str
    site_name: str
    resource_potential_tonnage: float
    high_prospectivity_area_ha: float
    next_7_day_production_mt: float
    shortfall_probability: float
    production_gap_mt: float
    critical_equipment_count: int
    recommendation_count: int
    model_health: str
    data_quality_score: float
    kpis: dict[str, Any]

