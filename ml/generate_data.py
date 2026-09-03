import os
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPACT = os.environ.get("MANGAI_COMPACT_DATA") == "1"


# ============================================================
# 1. GEOLOGICAL DATA
# ============================================================

def generate_geological_data(n=5000):

    print("Generating geological data...")

    latitude = np.random.uniform(21.20, 21.60, n)
    longitude = np.random.uniform(80.10, 80.50, n)

    elevation = np.random.normal(650, 70, n)

    slope = np.random.uniform(0, 35, n)

    aspect = np.random.uniform(0, 360, n)

    depth = np.random.uniform(0, 80, n)

    # Geological formations
    formations = np.random.choice(
        [
            "Gondite",
            "Quartzite",
            "Schist",
            "Laterite",
            "Granite",
            "Manganiferous_Formation"
        ],
        size=n,
        p=[0.20, 0.15, 0.15, 0.10, 0.15, 0.25]
    )

    # Base mineral concentrations
    mn_pct = np.random.normal(12, 7, n)
    fe_pct = np.random.normal(15, 5, n)
    sio2_pct = np.random.normal(35, 12, n)

    # Geological influence
    manganese_zone = formations == "Manganiferous_Formation"

    mn_pct[manganese_zone] += np.random.normal(18, 5, manganese_zone.sum())
    fe_pct[manganese_zone] -= np.random.normal(3, 1, manganese_zone.sum())
    sio2_pct[manganese_zone] -= np.random.normal(10, 4, manganese_zone.sum())

    # Depth influence
    mn_pct += np.where(
        (depth > 10) & (depth < 45),
        np.random.normal(3, 2, n),
        0
    )

    # Prevent impossible values
    mn_pct = np.clip(mn_pct, 0, 55)
    fe_pct = np.clip(fe_pct, 2, 40)
    sio2_pct = np.clip(sio2_pct, 5, 80)

    # Latent geological score
    geological_score = (
        0.45 * (mn_pct / 55)
        + 0.20 * (1 - sio2_pct / 80)
        + 0.15 * (fe_pct / 40)
        + 0.20 * manganese_zone.astype(float)
    )

    probability = 1 / (
        1 + np.exp(-8 * (geological_score - 0.45))
    )

    is_manganese = (
        np.random.random(n) < probability
    ).astype(int)

    # Ore thickness
    ore_thickness = np.where(
        is_manganese == 1,
        np.random.gamma(shape=3.5, scale=2.0, size=n),
        np.random.gamma(shape=1.5, scale=0.7, size=n)
    )

    ore_thickness = np.clip(ore_thickness, 0.1, 20)

    df = pd.DataFrame({
        "sample_id": [f"GS{i:05d}" for i in range(n)],
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "depth_m": depth,
        "formation": formations,
        "mn_pct": mn_pct,
        "fe_pct": fe_pct,
        "sio2_pct": sio2_pct,
        "ore_thickness_m": ore_thickness,
        "is_manganese": is_manganese
    })

    return df


# ============================================================
# 2. SATELLITE FEATURES
# ============================================================

def generate_satellite_data(geological_df):

    print("Generating satellite features...")

    n = len(geological_df)

    mn = geological_df["mn_pct"].values
    manganese = geological_df["is_manganese"].values

    # Spectral bands
    b2_blue = np.random.normal(0.12, 0.025, n)
    b3_green = np.random.normal(0.15, 0.03, n)
    b4_red = np.random.normal(0.16, 0.035, n)
    b8_nir = np.random.normal(0.35, 0.08, n)
    b11_swir = np.random.normal(0.28, 0.06, n)
    b12_swir = np.random.normal(0.22, 0.05, n)

    # Vegetation
    ndvi = (b8_nir - b4_red) / (
        b8_nir + b4_red + 1e-6
    )

    # Water index
    ndwi = (b3_green - b8_nir) / (
        b3_green + b8_nir + 1e-6
    )

    # SWIR ratio
    swir_ratio = b11_swir / (
        b12_swir + 1e-6
    )

    # Bare soil index
    bare_soil_index = (
        (b11_swir + b4_red)
        - (b8_nir + b2_blue)
    ) / (
        (b11_swir + b4_red)
        + (b8_nir + b2_blue)
        + 1e-6
    )

    # Land surface temperature
    lst = np.random.normal(31, 4, n)

    # Make exposed mineral zones slightly warmer
    lst += manganese * np.random.normal(1.5, 0.5, n)

    # Add a weak geological spectral signal
    spectral_signal = (
        manganese * np.random.normal(0.08, 0.02, n)
    )

    swir_ratio += spectral_signal

    df = pd.DataFrame({
        "sample_id": geological_df["sample_id"],
        "latitude": geological_df["latitude"],
        "longitude": geological_df["longitude"],
        "blue_b2": b2_blue,
        "green_b3": b3_green,
        "red_b4": b4_red,
        "nir_b8": b8_nir,
        "swir_b11": b11_swir,
        "swir_b12": b12_swir,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "swir_ratio": swir_ratio,
        "bare_soil_index": bare_soil_index,
        "land_surface_temperature": lst
    })

    return df


# ============================================================
# 3. BOREHOLE DATA
# ============================================================

def generate_borehole_data(n_boreholes=150):

    print("Generating borehole data...")

    records = []

    for i in range(n_boreholes):

        borehole_id = f"BH{i+1:04d}"

        latitude = np.random.uniform(21.20, 21.60)
        longitude = np.random.uniform(80.10, 80.50)

        total_depth = np.random.uniform(40, 180)

        current_depth = 0

        while current_depth < total_depth:

            interval = np.random.uniform(5, 15)

            end_depth = min(
                current_depth + interval,
                total_depth
            )

            lithology = np.random.choice(
                [
                    "Manganese_Ore",
                    "Manganiferous_Rock",
                    "Quartzite",
                    "Schist",
                    "Laterite",
                    "Waste"
                ],
                p=[
                    0.18,
                    0.17,
                    0.15,
                    0.15,
                    0.10,
                    0.25
                ]
            )

            if lithology == "Manganese_Ore":
                mn = np.random.normal(34, 6)
                fe = np.random.normal(11, 3)
                sio2 = np.random.normal(17, 5)

            elif lithology == "Manganiferous_Rock":
                mn = np.random.normal(20, 5)
                fe = np.random.normal(14, 4)
                sio2 = np.random.normal(25, 7)

            else:
                mn = np.random.normal(5, 3)
                fe = np.random.normal(18, 5)
                sio2 = np.random.normal(45, 12)

            records.append({
                "borehole_id": borehole_id,
                "latitude": latitude,
                "longitude": longitude,
                "from_depth_m": current_depth,
                "to_depth_m": end_depth,
                "lithology": lithology,
                "mn_pct": np.clip(mn, 0, 55),
                "fe_pct": np.clip(fe, 0, 45),
                "sio2_pct": np.clip(sio2, 0, 80)
            })

            current_depth = end_depth

    return pd.DataFrame(records)


# ============================================================
# 4. WEATHER DATA
# ============================================================

def generate_weather_data():

    print("Generating weather data...")

    dates = pd.date_range(
        start="2024-07-01" if COMPACT else "2023-01-01",
        end="2025-12-31" if not COMPACT else "2024-12-31",
        freq="D"
    )

    n = len(dates)

    month = dates.month.values

    # Monsoon influence
    monsoon_factor = np.isin(
        month,
        [6, 7, 8, 9]
    ).astype(float)

    rainfall = np.random.exponential(
        scale=2,
        size=n
    )

    rainfall += (
        monsoon_factor
        * np.random.exponential(8, n)
    )

    rainfall = np.clip(rainfall, 0, 180)

    soil_moisture = (
        0.25
        + rainfall / 300
        + np.random.normal(0, 0.04, n)
    )

    soil_moisture = np.clip(
        soil_moisture,
        0.05,
        0.95
    )

    temperature = (
        28
        + 6 * np.sin(
            2 * np.pi * (month - 1) / 12
        )
        + np.random.normal(0, 2, n)
    )

    vegetation_index = (
        0.35
        + 0.25 * monsoon_factor
        + np.random.normal(0, 0.05, n)
    )

    vegetation_index = np.clip(
        vegetation_index,
        0,
        1
    )

    return pd.DataFrame({
        "date": dates,
        "rainfall_mm": rainfall,
        "soil_moisture": soil_moisture,
        "temperature_c": temperature,
        "vegetation_index": vegetation_index
    })


# ============================================================
# 5. EQUIPMENT DATA
# ============================================================

def generate_equipment_data():

    print("Generating equipment data...")

    dates = pd.date_range(
        start="2024-07-01" if COMPACT else "2023-01-01",
        end="2025-12-31" if not COMPACT else "2024-12-31",
        freq="D"
    )

    equipment_types = [
        ("EX001", "Excavator", 100),
        ("EX002", "Excavator", 110),
        ("EX003", "Excavator", 95),
        ("DT001", "Dumper", 60),
        ("DT002", "Dumper", 65),
        ("DT003", "Dumper", 70),
        ("DR001", "Drill", 50),
        ("DR002", "Drill", 55),
    ]

    records = []

    for equipment_id, equipment_type, capacity in equipment_types:

        age_factor = np.random.uniform(
            0.8,
            1.2
        )

        for date in dates:

            operating_hours = np.clip(
                np.random.normal(8, 1.5),
                0,
                12
            )

            downtime = max(
                0,
                np.random.normal(
                    1.2 * age_factor,
                    1
                )
            )

            maintenance = (
                np.random.random() < 0.04
            )

            if maintenance:
                downtime += np.random.uniform(
                    5,
                    12
                )

            utilization = (
                operating_hours / 12
            )

            records.append({
                "date": date,
                "equipment_id": equipment_id,
                "equipment_type": equipment_type,
                "capacity_tph": capacity,
                "operating_hours": operating_hours,
                "downtime_hours": downtime,
                "utilization": utilization,
                "maintenance": int(maintenance)
            })

    return pd.DataFrame(records)


# ============================================================
# 6. BLASTING DATA
# ============================================================

def generate_blasting_data():

    print("Generating blasting data...")

    dates = pd.date_range(
        start="2024-07-01" if COMPACT else "2023-01-01",
        end="2025-12-31" if not COMPACT else "2024-12-31",
        freq="D"
    )

    n = len(dates)

    planned_blasts = np.random.poisson(
        1.2,
        n
    )

    planned_blasts = np.clip(
        planned_blasts,
        0,
        4
    )

    blasting_delay = np.random.exponential(
        1.5,
        n
    )

    delay_reason = np.random.choice(
        [
            "None",
            "Weather",
            "Equipment",
            "Safety_Clearance",
            "Material_Availability"
        ],
        n,
        p=[
            0.55,
            0.15,
            0.12,
            0.10,
            0.08
        ]
    )

    return pd.DataFrame({
        "date": dates,
        "planned_blasts": planned_blasts,
        "blasting_delay_hours": blasting_delay,
        "delay_reason": delay_reason
    })


# ============================================================
# 7. PRODUCTION DATA
# ============================================================

def generate_production_data(
    weather_df,
    equipment_df,
    blasting_df
):

    print("Generating production data...")

    dates = weather_df["date"]

    # Daily aggregate equipment information
    equipment_daily = (
        equipment_df
        .groupby("date")
        .agg({
            "operating_hours": "sum",
            "downtime_hours": "sum",
            "utilization": "mean"
        })
        .reset_index()
    )

    df = weather_df.merge(
        equipment_daily,
        on="date"
    )

    df = df.merge(
        blasting_df,
        on="date"
    )

    # Base production
    base_production = np.random.normal(
        8500,
        500,
        len(df)
    )

    # Weather impact
    weather_penalty = (
        df["rainfall_mm"] * 15
        + df["soil_moisture"] * 500
    )

    # Equipment impact
    equipment_penalty = (
        df["downtime_hours"] * 70
    )

    # Blasting impact
    blasting_penalty = (
        df["blasting_delay_hours"] * 120
    )

    production = (
        base_production
        - weather_penalty
        - equipment_penalty
        - blasting_penalty
    )

    production += np.random.normal(
        0,
        300,
        len(df)
    )

    production = np.clip(
        production,
        3000,
        None
    )

    target = np.random.normal(
        8500,
        200,
        len(df)
    )

    shortfall = (
        production < target
    ).astype(int)

    df["production_mt"] = production
    df["target_mt"] = target
    df["shortfall"] = shortfall

    # Production loss
    df["production_gap_mt"] = (
        df["target_mt"]
        - df["production_mt"]
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    geological = generate_geological_data(n=800 if COMPACT else 5000)

    satellite = generate_satellite_data(
        geological
    )

    boreholes = generate_borehole_data()

    weather = generate_weather_data()

    equipment = generate_equipment_data()

    blasting = generate_blasting_data()

    production = generate_production_data(
        weather,
        equipment,
        blasting
    )

    # Save datasets
    geological.to_csv(OUTPUT_DIR / "geological.csv", index=False)
    satellite.to_csv(OUTPUT_DIR / "satellite_features.csv", index=False)
    boreholes.to_csv(OUTPUT_DIR / "boreholes.csv", index=False)
    weather.to_csv(OUTPUT_DIR / "weather.csv", index=False)
    equipment.to_csv(OUTPUT_DIR / "equipment.csv", index=False)
    blasting.to_csv(OUTPUT_DIR / "blasting.csv", index=False)
    production.to_csv(OUTPUT_DIR / "production.csv", index=False)

    print("\n===================================")
    print("MANGAI DATA GENERATION COMPLETE")
    print("===================================")

    print(f"Geological records : {len(geological)}")
    print(f"Satellite records  : {len(satellite)}")
    print(f"Borehole records   : {len(boreholes)}")
    print(f"Weather records    : {len(weather)}")
    print(f"Equipment records  : {len(equipment)}")
    print(f"Blasting records   : {len(blasting)}")
    print(f"Production records : {len(production)}")

    print("\nFiles saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()