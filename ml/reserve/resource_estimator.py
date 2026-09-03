from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResourcePotentialResult:
    expected_tonnage: float
    p10: float
    p50: float
    p90: float


def estimate_resource_potential(
    probability: float,
    thickness_m: float,
    cell_area_m2: float = 10_000.0,
    density_t_per_m3: float = 3.6,
    probability_std: float = 0.08,
    thickness_std_fraction: float = 0.18,
    density_std: float = 0.18,
    samples: int = 1024,
    seed: int = 42,
) -> ResourcePotentialResult:
    rng = np.random.default_rng(seed)
    probability_samples = rng.normal(probability, probability_std, samples).clip(0.01, 0.99)
    thickness_samples = rng.normal(thickness_m, max(0.1, thickness_m * thickness_std_fraction), samples).clip(0.05, None)
    density_samples = rng.normal(density_t_per_m3, density_std, samples).clip(3.0, 4.2)
    tonnes = cell_area_m2 * thickness_samples * density_samples * probability_samples
    return ResourcePotentialResult(
        expected_tonnage=float(tonnes.mean()),
        p10=float(np.percentile(tonnes, 10)),
        p50=float(np.percentile(tonnes, 50)),
        p90=float(np.percentile(tonnes, 90)),
    )

