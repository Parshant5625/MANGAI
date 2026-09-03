from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.reserve.inference import predict_prospectivity_frame


def predict_prospectivity(geological_data: pd.DataFrame, satellite_data: pd.DataFrame, model_dir: Path | None = None) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    merged = geological_data.merge(satellite_data, on=["sample_id", "latitude", "longitude"], how="inner")
    return predict_prospectivity_frame(merged, model_dir or (root / "models"))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    geological = pd.read_csv(root / "data/synthetic/geological.csv")
    satellite = pd.read_csv(root / "data/synthetic/satellite_features.csv")
    predictions = predict_prospectivity(geological, satellite)
    output_path = root / "data/processed/reserve_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(predictions[["sample_id", "manganese_probability", "prospectivity_class"]].head())
    print(f"Predictions saved to: {output_path}")
