from __future__ import annotations

import pandas as pd


def ensure_columns(df: pd.DataFrame, columns: list[str], fill_value: float = 0.0) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = fill_value
    return output[columns]


def one_hot_align(df: pd.DataFrame, categorical_columns: list[str], feature_columns: list[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def spatial_block_id(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude", blocks: int = 5) -> pd.Series:
    lat_bins = pd.cut(df[lat_col], bins=blocks, labels=False, include_lowest=True)
    lon_bins = pd.cut(df[lon_col], bins=blocks, labels=False, include_lowest=True)
    return lat_bins.astype(str) + "_" + lon_bins.astype(str)

