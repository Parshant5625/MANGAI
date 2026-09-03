from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d",
    "rainfall_30d",
    "soil_moisture_1d",
    "temperature_c_1d",
    "vegetation_index_1d",
    "fleet_operating_hours_7d",
    "fleet_downtime_hours_7d",
    "fleet_utilization_7d",
    "planned_blasts_7d",
    "blasting_delay_7d",
    "production_lag_1",
    "production_lag_7",
    "production_lag_14",
    "production_lag_28",
    "production_mean_7",
    "production_mean_14",
    "production_mean_28",
    "downtime_mean_7",
    "rainfall_mean_7",
]


def build_daily_features(production: pd.DataFrame) -> pd.DataFrame:
    df = production.copy().sort_values("date").reset_index(drop=True)
    for window in [1, 3, 7, 30]:
        df[f"rainfall_{window}d"] = df["rainfall_mm"].rolling(window, min_periods=1).sum().shift(1)
    df["soil_moisture_1d"] = df.get("soil_moisture", 0.3).shift(1) if "soil_moisture" in df else 0.3
    df["temperature_c_1d"] = df.get("temperature_c", 27.0).shift(1) if "temperature_c" in df else 27.0
    if "vegetation_index" in df:
        df["vegetation_index_1d"] = df["vegetation_index"].shift(1)
    else:
        df["vegetation_index_1d"] = 0.4
    operating = df["operating_hours"] if "operating_hours" in df else pd.Series(0.0, index=df.index)
    utilization = df["utilization"] if "utilization" in df else pd.Series(0.65, index=df.index)
    planned = df["planned_blasts"] if "planned_blasts" in df else pd.Series(0.0, index=df.index)
    df["fleet_operating_hours_7d"] = operating.rolling(7, min_periods=1).sum().shift(1)
    df["fleet_downtime_hours_7d"] = df["downtime_hours"].rolling(7, min_periods=1).sum().shift(1)
    df["fleet_utilization_7d"] = utilization.rolling(7, min_periods=1).mean().shift(1)
    df["planned_blasts_7d"] = planned.rolling(7, min_periods=1).sum().shift(1)
    df["blasting_delay_7d"] = df["blasting_delay_hours"].rolling(7, min_periods=1).sum().shift(1)
    for lag in [1, 7, 14, 28]:
        df[f"production_lag_{lag}"] = df["production_mt"].shift(lag)
    for window in [7, 14, 28]:
        df[f"production_mean_{window}"] = df["production_mt"].rolling(window, min_periods=1).mean().shift(1)
    df["downtime_mean_7"] = df["downtime_hours"].rolling(7, min_periods=1).mean().shift(1)
    df["rainfall_mean_7"] = df["rainfall_mm"].rolling(7, min_periods=1).mean().shift(1)
    return df.dropna().reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(df) * train_fraction)
    validation_end = int(len(df) * (train_fraction + validation_fraction))
    return df.iloc[:train_end].copy(), df.iloc[train_end:validation_end].copy(), df.iloc[validation_end:].copy()
