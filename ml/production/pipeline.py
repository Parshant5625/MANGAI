from __future__ import annotations

from pathlib import Path

import pandas as pd

HORIZONS = (1, 7, 14, 28)
FORBIDDEN_FEATURES = {
    "production_mt",
    "production_gap_mt",
    "future_production_mt",
    "shortfall_label",
    "future_target_mt",
}


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "synthetic" / name, parse_dates=["date"])


def load_fused_production_data(root: Path) -> pd.DataFrame:
    """Build one row per date from production, weather, equipment and blasting.

    Equipment is aggregated before joining so the merge cannot create a
    many-to-many row explosion. The synthetic datasets have no site_id, so the
    caller's deterministic demo site is applied at the service boundary.
    """
    production = _read(root, "production.csv")
    weather = _read(root, "weather.csv")
    equipment = _read(root, "equipment.csv")
    blasting = _read(root, "blasting.csv")

    equipment_daily = (
        equipment.groupby("date", as_index=False)
        .agg(
            operating_hours=("operating_hours", "sum"),
            downtime_hours=("downtime_hours", "sum"),
            utilization=("utilization", "mean"),
            maintenance_events=("maintenance", "sum"),
            equipment_count=("equipment_id", "nunique"),
        )
    )
    frames = [production, weather, equipment_daily, blasting]
    result = frames[0].copy()
    for frame in frames[1:]:
        overlap = (set(result.columns) & set(frame.columns)) - {"date"}
        frame = frame.drop(columns=sorted(overlap), errors="ignore")
        result = result.merge(frame, on="date", how="left", validate="one_to_one")
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    return result.reset_index(drop=True)


def add_future_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon {horizon}; expected one of {HORIZONS}")
    out = df.sort_values("date").copy()
    out["future_production_mt"] = out["production_mt"].shift(-horizon)
    out["future_target_mt"] = out["target_mt"].shift(-horizon)
    out["future_gap_mt"] = out["future_production_mt"] - out["future_target_mt"]
    out["shortfall_label"] = (out["future_production_mt"] < out["future_target_mt"]).astype("int8")
    return out.dropna(subset=["future_production_mt", "future_target_mt"]).reset_index(drop=True)


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    leaked = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"target leakage detected in production features: {leaked}")


def chronological_splits(df: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be < 1")
    ordered = df.sort_values("date").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    if min(train_end, validation_end - train_end, len(ordered) - validation_end) < 1:
        raise ValueError("not enough rows for chronological split")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
