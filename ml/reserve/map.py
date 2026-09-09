import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "reserve_predictions.csv"
)


df = pd.read_csv(
    DATA_PATH
)


plt.figure(
    figsize=(10, 8)
)

scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["manganese_probability"],
    s=20,
    alpha=0.8
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "MANGAI — Manganese Prospectivity Map"
)

plt.colorbar(
    scatter,
    label="Manganese Probability"
)

plt.tight_layout()

output_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prospectivity_map.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print(
    f"Map saved to: {output_path}"
)