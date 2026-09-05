from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


def _validate_iso_date(value: str) -> str:
    date.fromisoformat(str(value))
    return str(value)


def _validate_iso_datetime(value: str) -> str:
    datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return str(value)


class DemoEnvelope(BaseModel):
    data_mode: Literal["demo", "live"] = "demo"
    synthetic_data: bool = True
    boundary_notice: str


class TopDriver(BaseModel):
    feature: str = Field(min_length=1, max_length=128)
    direction: Literal["positive", "negative", "neutral"]
    importance: float = Field(ge=0, le=1)
    value: float | str | None = None


class PredictionInterval(BaseModel):
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> PredictionInterval:
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("prediction interval must satisfy p10 <= p50 <= p90")
        return self


class ResourcePotential(BaseModel):
    label: str = "prototype resource potential"
    expected_tonnage: float = Field(ge=0)
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    assumptions: dict[str, Any]

    @model_validator(mode="after")
    def validate_order(self) -> ResourcePotential:
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("resource potential must satisfy p10 <= p50 <= p90")
        return self


class ProspectivityCell(BaseModel):
    id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    probability: float = Field(ge=0, le=1)
    prospectivity_class: Literal["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    predicted_grade_pct: float = Field(ge=0, le=100)
    predicted_thickness_m: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    resource_potential: ResourcePotential
    top_contributors: list[TopDriver]
    data_support: dict[str, Any] = Field(default_factory=dict)


class ReserveProspectivityResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    bbox: list[float] | None = None
    count: int = Field(ge=0)
    cells: list[ProspectivityCell]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("bbox must contain four coordinates")
        min_lon, min_lat, max_lon, max_lat = value
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("bbox longitude values must be between -180 and 180")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("bbox latitude values must be between -90 and 90")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox minimum coordinates must be smaller than maximum coordinates")
        return value


class ReserveSummaryResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    cells_evaluated: int = Field(ge=0)
    high_prospectivity_cells: int = Field(ge=0)
    very_high_prospectivity_cells: int = Field(ge=0)
    high_prospectivity_area_ha: float = Field(ge=0)
    average_probability: float = Field(ge=0, le=1)
    average_predicted_grade_pct: float = Field(ge=0, le=100)
    average_predicted_thickness_m: float = Field(ge=0)
    prototype_resource_potential: ResourcePotential
    validation_note: str


class ReservePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    elevation_m: float = Field(ge=-500, le=9000)
    slope_deg: float = Field(ge=0, le=90)
    aspect_deg: float = Field(ge=0, le=360)
    depth_m: float = Field(ge=0, le=3000)
    formation: str = Field(min_length=1, max_length=128)
    blue_b2: float = Field(default=0.12, ge=0, le=2)
    green_b3: float = Field(default=0.15, ge=0, le=2)
    red_b4: float = Field(default=0.16, ge=0, le=2)
    nir_b8: float = Field(default=0.35, ge=0, le=2)
    swir_b11: float = Field(default=0.28, ge=0, le=2)
    swir_b12: float = Field(default=0.22, ge=0, le=2)


class ReservePredictionResponse(DemoEnvelope):
    prediction: ProspectivityCell
    model_version: str = Field(min_length=1, max_length=128)


class ProductionForecastResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    forecast_date: str
    forecast_origin: str
    horizon_days: int = Field(ge=1, le=30)
    forecast_mt: float = Field(ge=0)
    target_mt: float = Field(gt=0)
    gap_mt: float
    shortfall_probability: float = Field(ge=0, le=1)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    prediction_interval: PredictionInterval
    top_drivers: list[TopDriver]
    model_version: str = Field(min_length=1, max_length=128)
    baseline_forecast_mt: float = Field(ge=0)
    data_freshness: dict[str, Any]
    horizon_series: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("forecast_date", "forecast_origin")
    @classmethod
    def validate_dates(cls, value: str) -> str:
        return _validate_iso_date(value)


class ProductionPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon_days: int = Field(default=7, ge=1, le=30)
    rainfall_mm_7d: float = Field(default=0, ge=0, le=5000)
    downtime_hours_7d: float = Field(default=0, ge=0, le=10000)
    blasting_delay_7d: float = Field(default=0, ge=0, le=10000)
    target_mt: float = Field(default=8500, gt=0, le=1_000_000)


class EquipmentItem(BaseModel):
    equipment_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    equipment_type: str = Field(min_length=1, max_length=64)
    availability: float = Field(ge=0, le=1)
    utilization: float = Field(ge=0, le=1.2)
    downtime_7d_hours: float = Field(ge=0)
    downtime_30d_hours: float = Field(ge=0)
    maintenance_events_30d: int = Field(ge=0)
    status: Literal["NORMAL", "WATCH", "CRITICAL"]


class EquipmentResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    fleet_availability: float = Field(ge=0, le=1)
    fleet_utilization: float = Field(ge=0, le=1.2)
    critical_equipment_count: int = Field(ge=0)
    items: list[EquipmentItem]
    maintenance_trend: list[dict[str, Any]] = Field(default_factory=list)


class WeatherResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    latest_date: str
    rainfall_7d_mm: float = Field(ge=0)
    rainfall_30d_mm: float = Field(ge=0)
    soil_moisture: float = Field(ge=0, le=1)
    temperature_c: float = Field(ge=-50, le=70)
    weather_risk: Literal["LOW", "MEDIUM", "HIGH"]
    observations: list[dict[str, Any]]

    @field_validator("latest_date")
    @classmethod
    def validate_latest_date(cls, value: str) -> str:
        return _validate_iso_date(value)


class BlastingResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    latest_date: str
    planned_blasts_7d: int = Field(ge=0)
    delay_hours_7d: float = Field(ge=0)
    delay_trend: Literal["IMPROVING", "STABLE", "WORSENING"]
    overlap_risk: Literal["LOW", "MEDIUM", "HIGH"]
    events: list[dict[str, Any]]

    @field_validator("latest_date")
    @classmethod
    def validate_latest_date(cls, value: str) -> str:
        return _validate_iso_date(value)


class Recommendation(BaseModel):
    id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    category: Literal["EQUIPMENT", "PRODUCTION", "WEATHER", "RESERVE", "BLASTING"]
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: dict[str, Any]
    estimated_impact: dict[str, Any]
    confidence: float = Field(ge=0, le=1)
    affected_equipment: list[str] = Field(default_factory=list)
    affected_area: str | None = None
    suggested_window: dict[str, Any]
    status: Literal["PROPOSED", "SIMULATED"] = "PROPOSED"


class RecommendationResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    recommendations: list[Recommendation]


class RecommendationSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reduce_downtime_pct: float = Field(default=0, ge=0, le=75)
    rainfall_risk_override: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    defer_weather_sensitive_blasts: bool = False


class ModelVersion(BaseModel):
    model_name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    task: str = Field(min_length=1, max_length=64)
    algorithm: str = Field(min_length=1, max_length=64)
    training_data_hash: str
    feature_schema_hash: str
    metrics: dict[str, Any]
    artifact_path: str
    created_at: str
    status: str = Field(min_length=1, max_length=64)
    notes: str = ""
    drift: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _validate_iso_datetime(value)


class ModelRegistryResponse(DemoEnvelope):
    models: list[ModelVersion]


class DataQualityRun(BaseModel):
    dataset_name: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=64)
    row_count: int = Field(ge=0)
    missing_rate: float = Field(ge=0, le=1)
    duplicate_rate: float = Field(ge=0, le=1)
    schema_valid: bool
    quality_score: float = Field(ge=0, le=1)
    details: dict[str, Any]
    created_at: str

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _validate_iso_datetime(value)


class DataQualityResponse(DemoEnvelope):
    overall_score: float = Field(ge=0, le=1)
    runs: list[DataQualityRun]


class OverviewResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    site_name: str = Field(min_length=1, max_length=128)
    resource_potential_tonnage: float = Field(ge=0)
    high_prospectivity_area_ha: float = Field(ge=0)
    next_7_day_production_mt: float = Field(ge=0)
    shortfall_probability: float = Field(ge=0, le=1)
    production_gap_mt: float
    critical_equipment_count: int = Field(ge=0)
    recommendation_count: int = Field(ge=0)
    model_health: Literal["READY", "WATCH"]
    data_quality_score: float = Field(ge=0, le=1)
    kpis: dict[str, Any]
