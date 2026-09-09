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
    "critical_equipment_count_7d",
    "maintenance_events_7d",
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


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def build_daily_features(production: pd.DataFrame) -> pd.DataFrame:
    """Create only information available at prediction time.

    Every contemporaneous operational/weather signal is shifted before it can
    enter a rolling statistic. Production targets and future targets are never
    feature columns.
    """
    df = production.copy().sort_values("date").reset_index(drop=True)
    rainfall = _series(df, "rainfall_mm")
    downtime = _series(df, "downtime_hours")
    operating = _series(df, "operating_hours")
    utilization = _series(df, "utilization", 0.65)
    planned = _series(df, "planned_blasts")
    blast_delay = _series(df, "blasting_delay_hours")
    maintenance = _series(df, "maintenance_events")
    equipment_count = _series(df, "equipment_count")
    production = _series(df, "production_mt")

    for window in (1, 3, 7, 30):
        df[f"rainfall_{window}d"] = rainfall.rolling(window, min_periods=window).sum().shift(1)
    df["soil_moisture_1d"] = _series(df, "soil_moisture", 0.3).shift(1)
    df["temperature_c_1d"] = _series(df, "temperature_c", 27.0).shift(1)
    df["vegetation_index_1d"] = _series(df, "vegetation_index", 0.4).shift(1)

    df["fleet_operating_hours_7d"] = operating.rolling(7, min_periods=7).sum().shift(1)
    df["fleet_downtime_hours_7d"] = downtime.rolling(7, min_periods=7).sum().shift(1)
    df["fleet_utilization_7d"] = utilization.rolling(7, min_periods=7).mean().shift(1)
    df["critical_equipment_count_7d"] = equipment_count.rolling(7, min_periods=7).mean().shift(1)
    df["maintenance_events_7d"] = maintenance.rolling(7, min_periods=7).sum().shift(1)
    df["planned_blasts_7d"] = planned.rolling(7, min_periods=7).sum().shift(1)
    df["blasting_delay_7d"] = blast_delay.rolling(7, min_periods=7).sum().shift(1)

    for lag in (1, 7, 14, 28):
        df[f"production_lag_{lag}"] = production.shift(lag)
    for window in (7, 14, 28):
        df[f"production_mean_{window}"] = production.rolling(window, min_periods=window).mean().shift(1)
    df["downtime_mean_7"] = downtime.rolling(7, min_periods=7).mean().shift(1)
    df["rainfall_mean_7"] = rainfall.rolling(7, min_periods=7).mean().shift(1)

    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(df) * train_fraction)
    validation_end = int(len(df) * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= len(df):
        raise ValueError("not enough rows for chronological split")
    ordered = df.sort_values("date").reset_index(drop=True)
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:validation_end].copy(), ordered.iloc[validation_end:].copy()
