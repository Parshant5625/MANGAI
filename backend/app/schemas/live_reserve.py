from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LiveSatelliteReserveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: str = Field(min_length=1, max_length=128)
    start: str = Field(min_length=10, max_length=10)
    end: str = Field(min_length=10, max_length=10)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    max_temporal_days: int = Field(default=16, ge=0, le=60)
    max_geology_distance_m: float = Field(default=500, gt=0, le=10_000)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("start", "end")
    @classmethod
    def validate_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_window(self) -> LiveSatelliteReserveRequest:
        if self.start > self.end:
            raise ValueError("start must not be after end")
        return self


class LiveReserveResourcePotential(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    expected_tonnage: float
    p10: float
    p50: float
    p90: float
    assumptions: dict[str, Any] = Field(default_factory=dict)
    uncertainty_sources: list[str] = Field(default_factory=list)


class LiveReserveDataSupport(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str
    satellite_source: str
    satellite_acquired_at: str | None = None
    satellite_provenance_checksum: str
    geology_source: str
    geology_match_distance_m: float
    feature_completeness: float
    model_version: str
    boundary: str


class LiveReserveCell(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    latitude: float
    longitude: float
    probability: float
    prospectivity_class: str
    predicted_grade_pct: float
    predicted_thickness_m: float
    confidence: float
    resource_potential: LiveReserveResourcePotential
    top_contributors: list[Any] = Field(default_factory=list)
    data_support: LiveReserveDataSupport


class LiveSatelliteReserveResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_mode: str
    synthetic_data: bool
    mixed_data: bool
    boundary_notice: str
    site_id: str
    count: int
    matched_geological_context: int
    unmatched_satellite: int
    geology_provenance: dict[str, Any]
    satellite_provenance: dict[str, Any]
    sentinel_scene_count: int
    thermal_scene_count: int
    temporal_distance_days: list[float]
    cells: list[LiveReserveCell]
