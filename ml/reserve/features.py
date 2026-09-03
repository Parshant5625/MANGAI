from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.preprocessing import one_hot_align


RESERVE_NUMERICAL_FEATURES = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "depth_m",
    "blue_b2",
    "green_b3",
    "red_b4",
    "nir_b8",
    "swir_b11",
    "swir_b12",
    "ndvi",
    "ndwi",
    "swir_ratio",
    "bare_soil_index",
    "land_surface_temperature",
]
RESERVE_CATEGORICAL_FEATURES = ["formation"]
LEAKAGE_EXCLUSIONS = ["mn_pct", "fe_pct", "sio2_pct", "is_manganese", "ore_thickness_m"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_fused_reserve_table(root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    geological = pd.read_csv(root / "data/synthetic/geological.csv")
    satellite = pd.read_csv(root / "data/synthetic/satellite_features.csv")
    return geological.merge(satellite, on=["sample_id", "latitude", "longitude"], how="inner")


def ensure_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "ndvi" not in output:
        output["ndvi"] = (output["nir_b8"] - output["red_b4"]) / (output["nir_b8"] + output["red_b4"] + 1e-6)
    if "ndwi" not in output:
        output["ndwi"] = (output["green_b3"] - output["nir_b8"]) / (output["green_b3"] + output["nir_b8"] + 1e-6)
    if "swir_ratio" not in output:
        output["swir_ratio"] = output["swir_b11"] / (output["swir_b12"] + 1e-6)
    if "bare_soil_index" not in output:
        output["bare_soil_index"] = ((output["swir_b11"] + output["red_b4"]) - (output["nir_b8"] + output["blue_b2"])) / (
            (output["swir_b11"] + output["red_b4"]) + (output["nir_b8"] + output["blue_b2"]) + 1e-6
        )
    if "land_surface_temperature" not in output:
        output["land_surface_temperature"] = output.get("lst_c", 31.0)
    return output


def prepare_reserve_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    prepared = ensure_spectral_indices(df)
    subset = prepared[RESERVE_NUMERICAL_FEATURES + RESERVE_CATEGORICAL_FEATURES].copy()
    if feature_columns is None:
        return pd.get_dummies(subset, columns=RESERVE_CATEGORICAL_FEATURES, dtype=int)
    return one_hot_align(subset, RESERVE_CATEGORICAL_FEATURES, feature_columns)
