from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DriftFeature(BaseModel):
    feature: str
    psi: float = Field(ge=0)
    status: str
    baseline_count: int = Field(ge=0)
    current_count: int = Field(ge=0)
    note: str = ""


class ModelMonitoringResponse(BaseModel):
    data_mode: str
    synthetic_data: bool
    checked_at: str
    registry: dict[str, Any]
    drift: list[DriftFeature]
    status: str
    alerts: list[dict[str, Any]] = Field(default_factory=list)
