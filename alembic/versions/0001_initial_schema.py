"""Initial MANGAI schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mine_sites",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("area_geojson", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_mine_sites_code", "mine_sites", ["code"], unique=True)

    op.create_table(
        "geological_samples",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("sample_id", sa.String(length=64), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("depth_m", sa.Float(), nullable=True),
        sa.Column("elevation_m", sa.Float(), nullable=True),
        sa.Column("slope_deg", sa.Float(), nullable=True),
        sa.Column("aspect_deg", sa.Float(), nullable=True),
        sa.Column("formation", sa.String(length=128), nullable=True),
        sa.Column("mn_pct", sa.Float(), nullable=True),
        sa.Column("fe_pct", sa.Float(), nullable=True),
        sa.Column("sio2_pct", sa.Float(), nullable=True),
        sa.Column("ore_thickness_m", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_geological_samples_sample_id", "geological_samples", ["sample_id"])

    op.create_table(
        "boreholes",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("borehole_id", sa.String(length=64), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("from_depth_m", sa.Float(), nullable=False),
        sa.Column("to_depth_m", sa.Float(), nullable=False),
        sa.Column("lithology", sa.String(length=128), nullable=True),
        sa.Column("mn_pct", sa.Float(), nullable=True),
        sa.Column("fe_pct", sa.Float(), nullable=True),
        sa.Column("sio2_pct", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_boreholes_borehole_id", "boreholes", ["borehole_id"])

    op.create_table(
        "satellite_observations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("blue_b2", sa.Float(), nullable=True),
        sa.Column("green_b3", sa.Float(), nullable=True),
        sa.Column("red_b4", sa.Float(), nullable=True),
        sa.Column("nir_b8", sa.Float(), nullable=True),
        sa.Column("swir_b11", sa.Float(), nullable=True),
        sa.Column("swir_b12", sa.Float(), nullable=True),
        sa.Column("ndvi", sa.Float(), nullable=True),
        sa.Column("ndwi", sa.Float(), nullable=True),
        sa.Column("swir_ratio", sa.Float(), nullable=True),
        sa.Column("bare_soil_index", sa.Float(), nullable=True),
        sa.Column("lst_c", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "weather_observations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rainfall_mm", sa.Float(), nullable=False),
        sa.Column("soil_moisture", sa.Float(), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=False),
        sa.Column("vegetation_index", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_weather_observations_observed_at", "weather_observations", ["observed_at"])

    op.create_table(
        "equipment",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("equipment_id", sa.String(length=64), nullable=False),
        sa.Column("equipment_type", sa.String(length=64), nullable=False),
        sa.Column("capacity_tph", sa.Float(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_equipment_equipment_id", "equipment", ["equipment_id"], unique=True)

    op.create_table(
        "equipment_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("equipment_id", sa.String(length=36), sa.ForeignKey("equipment.id"), nullable=False),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("operating_hours", sa.Float(), nullable=False),
        sa.Column("downtime_hours", sa.Float(), nullable=False),
        sa.Column("utilization", sa.Float(), nullable=False),
        sa.Column("maintenance", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_equipment_events_event_date", "equipment_events", ["event_date"])

    op.create_table(
        "blasting_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("planned_blasts", sa.Integer(), nullable=False),
        sa.Column("delay_hours", sa.Float(), nullable=False),
        sa.Column("delay_reason", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_blasting_events_event_date", "blasting_events", ["event_date"])

    op.create_table(
        "production_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("production_date", sa.Date(), nullable=False),
        sa.Column("actual_mt", sa.Float(), nullable=False),
        sa.Column("target_mt", sa.Float(), nullable=False),
        sa.Column("gap_mt", sa.Float(), nullable=False),
    )
    op.create_index("ix_production_records_production_date", "production_records", ["production_date"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("training_data_hash", sa.String(length=128), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("feature_schema", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model_version_id", sa.String(length=36), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("prediction_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=128), nullable=True),
        sa.Column("prediction", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "recommendations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("site_id", sa.String(length=36), sa.ForeignKey("mine_sites.id"), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("estimated_impact", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("suggested_window", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "data_quality_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_name", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("missing_rate", sa.Float(), nullable=False),
        sa.Column("duplicate_rate", sa.Float(), nullable=False),
        sa.Column("schema_valid", sa.Boolean(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in [
        "data_quality_runs",
        "recommendations",
        "predictions",
        "model_versions",
        "production_records",
        "blasting_events",
        "equipment_events",
        "equipment",
        "weather_observations",
        "satellite_observations",
        "boreholes",
        "geological_samples",
        "mine_sites",
    ]:
        op.drop_table(table)
