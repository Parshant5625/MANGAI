from __future__ import annotations

import math


def shortfall_probability(gap_mt: float, target_mt: float, risk_pressure: float = 0.0) -> float:
    shortage_ratio = max(0.0, -gap_mt / max(target_mt, 1.0))
    logit = -1.35 + 7.0 * shortage_ratio + risk_pressure
    return 1 / (1 + math.exp(-logit))


def severity(probability: float, gap_mt: float, target_mt: float) -> str:
    gap_ratio = max(0.0, -gap_mt / max(target_mt, 1.0))
    if probability >= 0.8 or gap_ratio >= 0.18:
        return "CRITICAL"
    if probability >= 0.65 or gap_ratio >= 0.1:
        return "HIGH"
    if probability >= 0.4 or gap_ratio >= 0.04:
        return "MEDIUM"
    return "LOW"

