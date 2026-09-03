from ml.common.preprocessing import spatial_block_id
from ml.production.features import FEATURE_COLUMNS, build_daily_features
import pandas as pd


def test_spatial_blocks_keep_groups(demo_store):
    df = demo_store.geological()
    blocks = spatial_block_id(df)
    assert blocks.nunique() > 1
    assert len(blocks) == len(df)


def test_production_features_are_lagged(demo_store):
    features = build_daily_features(demo_store.production())
    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert features["production_lag_1"].notna().all()
