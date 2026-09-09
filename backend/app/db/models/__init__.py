from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


def uuid_text() -> str:
    return str(uuid4())


class MineSite(Base):
    __tablename__ = "mine_sites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    name: Mapped[str] = mapped_column(Text)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_geojson: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class GeologicalSample(Base):
    __tablename__ = "geological_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    sample_id: Mapped[str] = mapped_column(String(64), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    depth_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    slope_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    aspect_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    formation: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mn_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fe_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    sio2_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ore_thickness_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="demo_synthetic")
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BoreholeInterval(Base):
    __tablename__ = "boreholes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    borehole_id: Mapped[str] = mapped_column(String(64), index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    from_depth_m: Mapped[float] = mapped_column(Float)
    to_depth_m: Mapped[float] = mapped_column(Float)
    lithology: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mn_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fe_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    sio2_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="demo_synthetic")


class SatelliteObservation(Base):
    __tablename__ = "satellite_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    blue_b2: Mapped[float | None] = mapped_column(Float, nullable=True)
    green_b3: Mapped[float | None] = mapped_column(Float, nullable=True)
    red_b4: Mapped[float | None] = mapped_column(Float, nullable=True)
    nir_b8: Mapped[float | None] = mapped_column(Float, nullable=True)
    swir_b11: Mapped[float | None] = mapped_column(Float, nullable=True)
    swir_b12: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndwi: Mapped[float | None] = mapped_column(Float, nullable=True)
    swir_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    bare_soil_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    lst_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="demo_synthetic")


class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    rainfall_mm: Mapped[float] = mapped_column(Float)
    soil_moisture: Mapped[float] = mapped_column(Float)
    temperature_c: Mapped[float] = mapped_column(Float)
    vegetation_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="demo_synthetic")


class Equipment(Base):
    __tablename__ = "equipment"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    equipment_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    equipment_type: Mapped[str] = mapped_column(String(64))
    capacity_tph: Mapped[float] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class EquipmentEvent(Base):
    __tablename__ = "equipment_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    equipment_id: Mapped[str] = mapped_column(String(36), ForeignKey("equipment.id"))
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    operating_hours: Mapped[float] = mapped_column(Float)
    downtime_hours: Mapped[float] = mapped_column(Float)
    utilization: Mapped[float] = mapped_column(Float)
    maintenance: Mapped[bool] = mapped_column(Boolean, default=False)


class BlastingEvent(Base):
    __tablename__ = "blasting_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    planned_blasts: Mapped[int] = mapped_column(Integer)
    delay_hours: Mapped[float] = mapped_column(Float)
    delay_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ProductionRecord(Base):
    __tablename__ = "production_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    production_date: Mapped[date] = mapped_column(Date, index=True)
    actual_mt: Mapped[float] = mapped_column(Float)
    target_mt: Mapped[float] = mapped_column(Float)
    gap_mt: Mapped[float] = mapped_column(Float)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(64))
    training_data_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(64), default="candidate")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    model_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    prediction_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prediction: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    site_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("mine_sites.id"), nullable=True)
    category: Mapped[str] = mapped_column(String(64))
    priority: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JSON)
    estimated_impact: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    suggested_window: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PROPOSED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class DataQualityRun(Base):
    __tablename__ = "data_quality_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_text)
    dataset_name: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(64))
    row_count: Mapped[int] = mapped_column(Integer)
    missing_rate: Mapped[float] = mapped_column(Float)
    duplicate_rate: Mapped[float] = mapped_column(Float)
    schema_valid: Mapped[bool] = mapped_column(Boolean)
    quality_score: Mapped[float] = mapped_column(Float)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

