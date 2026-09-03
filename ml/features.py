import pandas as pd
from pathlib import Path


def load_geological_data():
    """
    Load the geological dataset.
    """

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "synthetic"
        / "geological.csv"
    )

    df = pd.read_csv(data_path)

    return df


def prepare_features():
    """
    Prepare features (X) and target (y)
    for the manganese prospectivity model.
    """

    df = load_geological_data()

    # Features used by the ML model
    feature_columns = [
        "latitude",
        "longitude",
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "depth_m",
        "mn_pct",
        "fe_pct",
        "sio2_pct",
        "ore_thickness_m"
    ]

    # Input data
    X = df[feature_columns].copy()

    # Target: manganese occurrence
    y = df["is_manganese"].copy()

    return X, y


if __name__ == "__main__":

    X, y = prepare_features()

    print("Feature preparation successful!")
    print()
    print("Features:")
    print(X.head())
    print()
    print("Feature shape:", X.shape)
    print()
    print("Target distribution:")
    print(y.value_counts())
    