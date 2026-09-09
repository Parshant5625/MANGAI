from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.schemas.api import DemoEnvelope


class OperationsAssociation(BaseModel):
    driver: Literal["EQUIPMENT_DOWNTIME", "BLASTING_DELAY", "RAINFALL"]
    metric: str = Field(min_length=1, max_length=128)
    correlation: float = Field(ge=-1, le=1)
    sample_size: int = Field(ge=0)
    direction: Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "INSUFFICIENT_DATA"]
    interpretation: str = Field(min_length=1)


class OperationsRiskSignal(BaseModel):
    source: Literal["EQUIPMENT", "WEATHER", "BLASTING", "CROSS_DOMAIN"]
    level: Literal["LOW", "MEDIUM", "HIGH"]
    score: float = Field(ge=0, le=1)
    title: str = Field(min_length=1)
    evidence: dict[str, Any]


class OperationsSummaryResponse(DemoEnvelope):
    site_id: str = Field(min_length=1, max_length=128)
    latest_date: str
    analysis_window_days: int = Field(ge=7, le=365)
    production_records: int = Field(ge=0)
    production_mt_mean: float = Field(ge=0)
    target_mt_mean: float = Field(ge=0)
    gap_mt_mean: float
    fleet_availability: float = Field(ge=0, le=1)
    fleet_utilization: float = Field(ge=0, le=1.2)
    rainfall_7d_mm: float = Field(ge=0)
    soil_moisture: float = Field(ge=0, le=1)
    planned_blasts_7d: int = Field(ge=0)
    blasting_delay_7d_hours: float = Field(ge=0)
    production_associations: list[OperationsAssociation]
    risk_signals: list[OperationsRiskSignal]
    overall_operational_risk: Literal["LOW", "MEDIUM", "HIGH"]
    overall_risk_score: float = Field(ge=0, le=1)
    data_coverage: dict[str, Any]
    methodology_note: str = Field(min_length=1)
