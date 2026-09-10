from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        from datetime import date

        date.fromisoformat(value)
        return value

    def validate_window(self) -> None:
        if self.start > self.end:
            raise ValueError("start must not be after end")
