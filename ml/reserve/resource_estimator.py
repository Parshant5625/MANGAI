from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResourcePotentialResult:
    expected_tonnage: float
    p10: float
    p50: float
    p90: float
    assumptions: dict[str, float] | None = None
    uncertainty_sources: dict[str, bool] | None = None


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


def estimate_resource_potential_with_intervals(
    probability: float,
    thickness_point: float,
    thickness_low: float,
    thickness_high: float,
    cell_area_m2: float = 10_000.0,
    density_t_per_m3: float = 3.6,
    *,
    probability_std: float | None = None,
    density_std: float | None = None,
    samples: int = 2048,
    seed: int = 42,
) -> ResourcePotentialResult:
    """Propagate uncertainty from a conformal thickness interval.

    Only model-derived uncertainty is propagated: the thickness prediction
    interval directly determines a thickness distribution (a normal whose
    5th/95th percentiles match the conformal lower/upper bounds). The density
    assumption remains configurable and, per the project boundary, NO invented
    geological density distribution is used — if ``density_std`` is omitted,
    density is treated as a fixed assumption and that is reported explicitly.
    """
    rng = np.random.default_rng(seed)
    z90 = 1.6449  # 90% central interval
    half_width = max((thickness_high - thickness_low) / 2.0, 0.05)
    thickness_std = half_width / z90
    thickness_mean = np.clip((thickness_low + thickness_high) / 2.0, 0.05, None)
    thickness_samples = rng.normal(thickness_mean, thickness_std, samples).clip(0.05, None)

    if probability_std is None:
        probability_samples = np.full(samples, np.clip(probability, 0.01, 0.99))
    else:
        probability_samples = rng.normal(probability, probability_std, samples).clip(0.01, 0.99)

    if density_std is None:
        density_samples = np.full(samples, density_t_per_m3)
    else:
        density_samples = rng.normal(density_t_per_m3, density_std, samples).clip(3.0, 4.2)

    tonnes = cell_area_m2 * thickness_samples * density_samples * probability_samples
    return ResourcePotentialResult(
        expected_tonnage=float(tonnes.mean()),
        p10=float(np.percentile(tonnes, 10)),
        p50=float(np.percentile(tonnes, 50)),
        p90=float(np.percentile(tonnes, 90)),
        assumptions={
            "cell_area_m2": cell_area_m2,
            "density_t_per_m3": density_t_per_m3,
            "uncertainty_method": "Monte Carlo from conformal thickness interval",
            "classification_boundary": "prototype only, not official reserves",
        },
        uncertainty_sources={
            "probability": probability_std is not None,
            "thickness": True,
            "density": density_std is not None,
        },
    )

