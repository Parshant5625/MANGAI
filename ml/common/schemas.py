from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    features: list[str]
    target: str | None
    leakage_exclusions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": self.features,
            "target": self.target,
            "leakage_exclusions": self.leakage_exclusions,
        }

