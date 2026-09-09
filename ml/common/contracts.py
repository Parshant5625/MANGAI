"""Canonical data contracts for MANGAI datasets.

Every major dataset in the MANGAI pipeline is described by a ColumnContract
and grouped under a DatasetContract. These contracts are machine-readable,
validated at runtime, and kept in one place so that generation, validation,
seeding, and feature engineering all share the same source of truth.

Units are documented explicitly. All synthetic/demo data is clearly labeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ColumnContract:
    """Description of a single dataset column."""

    name: str
    dtype: str  # "str", "float", "int", "date", "datetime"
    unit: str = ""
    description: str = ""
    required: bool = True
    nullable: bool = False
    unique: bool = False
    min_value: float | None = None
    max_value: float | None = None
    categorical_values: tuple[str, ...] = ()
    synthetic_default: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "unit": self.unit,
            "description": self.description,
            "required": self.required,
            "nullable": self.nullable,
            "unique": self.unique,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "categorical_values": list(self.categorical_values),
        }


@dataclass(frozen=True)
class DatasetContract:
    """Description of an entire dataset."""

    name: str
    description: str
    columns: tuple[ColumnContract, ...]
    source: str = "demo_synthetic"
    primary_key: str = ""

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def required_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.required]

    def column(self, name: str) -> ColumnContract:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Column {name!r} not in contract {self.name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "primary_key": self.primary_key,
            "columns": [column.to_dict() for column in self.columns],
        }


# ============================================================
# A. Geological samples
# ============================================================

GEOLOGICAL_CONTRACT = DatasetContract(
    name="geological",
    description="Synthetic geological sample observations for manganese prospectivity modeling.",
    primary_key="sample_id",
    columns=(
        ColumnContract("sample_id", "str", unique=True, description="Sample identifier (GS00000 format)."),
        ColumnContract("latitude", "float", unit="decimal degrees", min_value=-90.0, max_value=90.0, description="WGS84 latitude."),
        ColumnContract("longitude", "float", unit="decimal degrees", min_value=-180.0, max_value=180.0, description="WGS84 longitude."),
        ColumnContract("elevation_m", "float", unit="meters", min_value=-500.0, max_value=9000.0, description="Surface elevation above sea level."),
        ColumnContract("slope_deg", "float", unit="degrees", min_value=0.0, max_value=90.0, description="Terrain slope."),
        ColumnContract("aspect_deg", "float", unit="degrees", min_value=0.0, max_value=360.0, description="Terrain aspect (compass direction)."),
        ColumnContract("depth_m", "float", unit="meters", min_value=0.0, max_value=2000.0, description="Sample depth below surface."),
        ColumnContract(
            "formation",
            "str",
            description="Geological formation name.",
            categorical_values=(
                "Gondite",
                "Quartzite",
                "Schist",
                "Laterite",
                "Granite",
                "Manganiferous_Formation",
            ),
        ),
        ColumnContract("mn_pct", "float", unit="percent", min_value=0.0, max_value=60.0, description="Manganese grade (TARGET - leakage-sensitive)."),
        ColumnContract("fe_pct", "float", unit="percent", min_value=0.0, max_value=60.0, description="Iron grade (TARGET - leakage-sensitive)."),
        ColumnContract("sio2_pct", "float", unit="percent", min_value=0.0, max_value=100.0, description="Silica grade (TARGET - leakage-sensitive)."),
        ColumnContract("ore_thickness_m", "float", unit="meters", min_value=0.0, max_value=50.0, description="Ore thickness (TARGET - leakage-sensitive)."),
        ColumnContract("is_manganese", "int", unit="binary indicator", min_value=0, max_value=1, description="Binary manganese indicator (TARGET - leakage-sensitive)."),
    ),
)

# ============================================================
# B. Boreholes
# ============================================================

BOREHOLE_CONTRACT = DatasetContract(
    name="boreholes",
    description="Synthetic borehole interval records with lithology and assay results. "
    "Each row is one interval; borehole_id groups intervals belonging to the same collar.",
    columns=(
        ColumnContract("borehole_id", "str", description="Borehole identifier (shared by all intervals of one borehole)."),
        ColumnContract("latitude", "float", unit="decimal degrees", min_value=-90.0, max_value=90.0, description="WGS84 latitude."),
        ColumnContract("longitude", "float", unit="decimal degrees", min_value=-180.0, max_value=180.0, description="WGS84 longitude."),
        ColumnContract("from_depth_m", "float", unit="meters", min_value=0.0, description="Interval start depth."),
        ColumnContract("to_depth_m", "float", unit="meters", min_value=0.0, description="Interval end depth."),
        ColumnContract("lithology", "str", description="Lithology description."),
        ColumnContract("mn_pct", "float", unit="percent", min_value=0.0, max_value=60.0, description="Manganese assay."),
        ColumnContract("fe_pct", "float", unit="percent", min_value=0.0, max_value=60.0, description="Iron assay."),
        ColumnContract("sio2_pct", "float", unit="percent", min_value=0.0, max_value=100.0, description="Silica assay."),
    ),
)

# ============================================================
# C. Satellite observations / features
# ============================================================

SATELLITE_CONTRACT = DatasetContract(
    name="satellite_features",
    description="Synthetic satellite spectral bands and derived indices aligned to geological sample locations.",
    primary_key="sample_id",
    columns=(
        ColumnContract("sample_id", "str", unique=True, description="Sample identifier (links to geological.sample_id)."),
        ColumnContract("latitude", "float", unit="decimal degrees", min_value=-90.0, max_value=90.0, description="WGS84 latitude."),
        ColumnContract("longitude", "float", unit="decimal degrees", min_value=-180.0, max_value=180.0, description="WGS84 longitude."),
        ColumnContract("blue_b2", "float", unit="reflectance", min_value=0.0, max_value=1.0, description="Blue band (B2)."),
        ColumnContract("green_b3", "float", unit="reflectance", min_value=0.0, max_value=1.0, description="Green band (B3)."),
        ColumnContract("red_b4", "float", unit="reflectance", min_value=0.0, max_value=1.0, description="Red band (B4)."),
        ColumnContract("nir_b8", "float", unit="reflectance", min_value=0.0, max_value=1.0, description="Near-infrared band (B8)."),
        ColumnContract("swir_b11", "float", unit="reflectance", min_value=0.0, max_value=1.0, description="Short-wave infrared band (B11)."),
        ColumnContract("swir_b12", "float", unit="reflectance", min_value=0.0, max_value=1.0, description="Short-wave infrared band (B12)."),
        ColumnContract("ndvi", "float", unit="index", min_value=-1.0, max_value=1.0, description="Normalized Difference Vegetation Index."),
        ColumnContract("ndwi", "float", unit="index", min_value=-1.0, max_value=1.0, description="Normalized Difference Water Index."),
        ColumnContract("swir_ratio", "float", unit="ratio", min_value=0.0, description="SWIR B11/B12 ratio."),
        ColumnContract("bare_soil_index", "float", unit="index", min_value=-1.0, max_value=1.0, description="Bare soil index."),
        ColumnContract("land_surface_temperature", "float", unit="degrees Celsius", min_value=-50.0, max_value=60.0, description="Land surface temperature."),
    ),
)

# ============================================================
# D. Weather observations
# ============================================================

WEATHER_CONTRACT = DatasetContract(
    name="weather",
    description="Synthetic daily weather observations.",
    primary_key="date",
    columns=(
        ColumnContract("date", "date", description="Observation date (YYYY-MM-DD)."),
        ColumnContract("rainfall_mm", "float", unit="millimeters", min_value=0.0, max_value=500.0, description="Daily rainfall."),
        ColumnContract("soil_moisture", "float", unit="fraction", min_value=0.0, max_value=1.0, description="Volumetric soil moisture."),
        ColumnContract("temperature_c", "float", unit="degrees Celsius", min_value=-40.0, max_value=55.0, description="Air temperature."),
        ColumnContract("vegetation_index", "float", unit="index", min_value=-1.0, max_value=1.0, description="Vegetation index (e.g. NDVI)."),
    ),
)

# ============================================================
# E. Equipment
# ============================================================

EQUIPMENT_CONTRACT = DatasetContract(
    name="equipment",
    description="Synthetic daily equipment telemetry and maintenance records.",
    columns=(
        ColumnContract("date", "date", description="Record date (YYYY-MM-DD)."),
        ColumnContract("equipment_id", "str", description="Equipment identifier."),
        ColumnContract(
            "equipment_type",
            "str",
            description="Equipment category.",
            categorical_values=("Excavator", "Haul_Truck", "Drill", "Loader", "Dozer"),
        ),
        ColumnContract("capacity_tph", "float", unit="tons per hour", min_value=0.0, description="Nominal capacity."),
        ColumnContract("operating_hours", "float", unit="hours", min_value=0.0, max_value=24.0, description="Daily operating hours."),
        ColumnContract("downtime_hours", "float", unit="hours", min_value=0.0, max_value=24.0, description="Daily downtime hours."),
        ColumnContract("utilization", "float", unit="fraction", min_value=0.0, max_value=1.5, description="Utilization ratio."),
        ColumnContract("maintenance", "int", unit="binary indicator", min_value=0, max_value=1, description="Maintenance event flag."),
    ),
)

# ============================================================
# F. Equipment events (derived)
# ============================================================

EQUIPMENT_EVENT_CONTRACT = DatasetContract(
    name="equipment_events",
    description="Synthetic equipment event records derived from daily equipment telemetry.",
    columns=(
        ColumnContract("equipment_id", "str", description="Equipment identifier (links to equipment.equipment_id)."),
        ColumnContract("event_date", "date", description="Event date (YYYY-MM-DD)."),
        ColumnContract("operating_hours", "float", unit="hours", min_value=0.0, max_value=24.0, description="Operating hours for the event."),
        ColumnContract("downtime_hours", "float", unit="hours", min_value=0.0, max_value=24.0, description="Downtime hours for the event."),
        ColumnContract("utilization", "float", unit="fraction", min_value=0.0, max_value=1.5, description="Utilization ratio."),
        ColumnContract("maintenance", "int", unit="binary indicator", min_value=0, max_value=1, description="Maintenance event flag."),
    ),
)

# ============================================================
# G. Blasting events
# ============================================================

BLASTING_CONTRACT = DatasetContract(
    name="blasting",
    description="Synthetic daily blasting event records.",
    primary_key="date",
    columns=(
        ColumnContract("date", "date", description="Event date (YYYY-MM-DD)."),
        ColumnContract("planned_blasts", "int", unit="count", min_value=0, description="Number of planned blasts."),
        ColumnContract("blasting_delay_hours", "float", unit="hours", min_value=0.0, description="Total blasting delay."),
        ColumnContract(
            "delay_reason",
            "str",
            description="Reason for blasting delay. 'No_Delay' means blasting proceeded on schedule. "
            "(Never persist the literal value 'None': pandas read_csv parses it as NaN.)",
            categorical_values=("No_Delay", "Weather", "Equipment", "Safety", "Logistics"),
        ),
    ),
)

# ============================================================
# H. Production records
# ============================================================

PRODUCTION_CONTRACT = DatasetContract(
    name="production",
    description="Synthetic daily production records with operational context "
    "(weather, fleet, and blasting columns are carried for feature engineering).",
    primary_key="date",
    columns=(
        ColumnContract("date", "date", description="Production date (YYYY-MM-DD)."),
        ColumnContract("production_mt", "float", unit="metric tons", min_value=0.0, description="Actual production (TARGET - leakage-sensitive)."),
        ColumnContract("target_mt", "float", unit="metric tons", min_value=0.0, description="Production target (TARGET - leakage-sensitive)."),
        ColumnContract("production_gap_mt", "float", unit="metric tons", description="Target minus actual (positive = shortfall) (TARGET - leakage-sensitive)."),
        ColumnContract("rainfall_mm", "float", unit="millimeters", min_value=0.0, description="Daily rainfall."),
        ColumnContract("soil_moisture", "float", unit="fraction", min_value=0.0, max_value=1.0, description="Volumetric soil moisture."),
        ColumnContract("temperature_c", "float", unit="degrees Celsius", min_value=-40.0, max_value=55.0, description="Air temperature."),
        ColumnContract("vegetation_index", "float", unit="index", min_value=-1.0, max_value=1.0, description="Vegetation index."),
        ColumnContract("operating_hours", "float", unit="hours", min_value=0.0, description="Fleet operating hours."),
        ColumnContract("downtime_hours", "float", unit="hours", min_value=0.0, description="Fleet downtime hours."),
        ColumnContract("utilization", "float", unit="fraction", min_value=0.0, max_value=1.5, description="Fleet utilization ratio."),
        ColumnContract("planned_blasts", "int", unit="count", min_value=0, description="Planned blasts on the date."),
        ColumnContract("blasting_delay_hours", "float", unit="hours", min_value=0.0, description="Blasting delay hours."),
    ),
)


# ============================================================
# Registry
# ============================================================

ALL_CONTRACTS: dict[str, DatasetContract] = {
    contract.name: contract
    for contract in (
        GEOLOGICAL_CONTRACT,
        BOREHOLE_CONTRACT,
        SATELLITE_CONTRACT,
        WEATHER_CONTRACT,
        EQUIPMENT_CONTRACT,
        EQUIPMENT_EVENT_CONTRACT,
        BLASTING_CONTRACT,
        PRODUCTION_CONTRACT,
    )
}

# Columns that are prediction targets and must never appear in a feature matrix.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "mn_pct",
        "fe_pct",
        "sio2_pct",
        "ore_thickness_m",
        "is_manganese",
        "production_mt",
        "target_mt",
        "production_gap_mt",
        "shortfall",
    }
)
