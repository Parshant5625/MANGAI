from __future__ import annotations


def shortfall_probability(gap_mt: float, target_mt: float, risk_pressure: float = 0.0) -> float:
    """Fallback-only probability used when a trained classifier is unavailable."""
    shortage_ratio = max(0.0, -gap_mt / max(target_mt, 1.0))
    logit = -1.35 + 7.0 * shortage_ratio + risk_pressure
    import math

    return 1.0 / (1.0 + math.exp(-logit))


def severity(probability: float, gap_mt: float, target_mt: float) -> str:
    """Operational severity: shortfall ratio bands are 0–5, 5–10, 10–20, >20%."""
    gap_ratio = max(0.0, -gap_mt / max(target_mt, 1.0))
    if probability >= 0.80 or gap_ratio > 0.20:
        return "CRITICAL"
    if probability >= 0.65 or gap_ratio > 0.10:
        return "HIGH"
    if probability >= 0.40 or gap_ratio > 0.05:
        return "MEDIUM"
    return "LOW"
