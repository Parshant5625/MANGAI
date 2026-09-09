from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope


class OperationsSummaryService:
    """Cross-domain operational analytics.

    This service is deliberately descriptive/associational. Correlation is not
    presented as a causal production impact and no recommendation is generated here.
    All joins are reduced to one row per date before cross-domain merging.
    """

    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    @staticmethod
    def _one_site(df: pd.DataFrame, site_id: str | None) -> pd.DataFrame:
        if site_id and "site_id" in df.columns:
            scoped = df[df["site_id"].astype(str) == str(site_id)].copy()
            return scoped
        return df.copy()

    @staticmethod
    def _daily_numeric(df: pd.DataFrame, date_col: str, value_cols: list[str], agg: str = "mean") -> pd.DataFrame:
        if df.empty or date_col not in df.columns:
            return pd.DataFrame(columns=["date", *value_cols])
        work = df.copy()
        work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
        work = work.dropna(subset=["date"])
        for col in value_cols:
            if col not in work.columns:
                work[col] = np.nan
            work[col] = pd.to_numeric(work[col], errors="coerce")
        if agg == "sum":
            return work.groupby("date", as_index=False)[value_cols].sum(min_count=1)
        return work.groupby("date", as_index=False)[value_cols].mean()

    @staticmethod
    def _corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
        pair = pd.concat([x, y], axis=1).dropna()
        n = len(pair)
        if n < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            return 0.0, n
        value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        return (value if np.isfinite(value) else 0.0), n

    @staticmethod
    def _direction(correlation: float, n: int) -> str:
        if n < 3:
            return "INSUFFICIENT_DATA"
        if correlation >= 0.15:
            return "POSITIVE"
        if correlation <= -0.15:
            return "NEGATIVE"
        return "NEUTRAL"

    @classmethod
    def _association(cls, frame: pd.DataFrame, driver: str, metric: str, column: str) -> dict[str, Any]:
        corr, n = cls._corr(frame[column], frame["production_mt"])
        direction = cls._direction(corr, n)
        if n < 3:
            interpretation = "Insufficient aligned observations to estimate an association."
        elif direction == "POSITIVE":
            interpretation = f"Higher {metric.lower()} is associated with higher observed production in this window."
        elif direction == "NEGATIVE":
            interpretation = f"Higher {metric.lower()} is associated with lower observed production in this window."
        else:
            interpretation = f"No material linear association between {metric.lower()} and observed production in this window."
        return {
            "driver": driver,
            "metric": metric,
            "correlation": round(corr, 4),
            "sample_size": n,
            "direction": direction,
            "interpretation": interpretation,
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.67:
            return "HIGH"
        if score >= 0.34:
            return "MEDIUM"
        return "LOW"

    def summary(self, site_id: str | None = None, days: int = 30) -> dict[str, Any]:
        days = max(7, min(int(days), 365))
        production = self._one_site(self.store.production(), site_id)
        weather = self._one_site(self.store.weather(), site_id)
        equipment = self._one_site(self.store.equipment(), site_id)
        blasting = self._one_site(self.store.blasting(), site_id)

        production = self._daily_numeric(production, "date", ["production_mt", "target_mt"], "mean")
        production["production_mt"] = production["production_mt"].fillna(0.0)
        production["target_mt"] = production["target_mt"].fillna(0.0)
        production["gap_mt"] = production["production_mt"] - production["target_mt"]
        latest = production["date"].max()
        if pd.isna(latest):
            raise ValueError("production dataset contains no valid dates")
        start = latest - pd.Timedelta(days=days - 1)
        production = production[production["date"] >= start].copy()

        weather_d = self._daily_numeric(weather, "date", ["rainfall_mm", "soil_moisture", "temperature_c"], "mean")
        equipment_d = self._daily_numeric(equipment, "date", ["downtime_hours", "utilization", "maintenance"], "sum")
        if not equipment_d.empty:
            util = self._daily_numeric(equipment, "date", ["utilization"], "mean")
            equipment_d = equipment_d.drop(columns=["utilization"], errors="ignore").merge(util, on="date", how="left", validate="one_to_one")
        blasting_d = self._daily_numeric(blasting, "date", ["blasting_delay_hours", "planned_blasts"], "sum")

        merged = production.merge(weather_d, on="date", how="left", validate="one_to_one")
        merged = merged.merge(equipment_d, on="date", how="left", validate="one_to_one")
        merged = merged.merge(blasting_d, on="date", how="left", validate="one_to_one")
        merged = merged.sort_values("date").reset_index(drop=True)

        associations = [
            self._association(merged, "EQUIPMENT_DOWNTIME", "equipment downtime hours", "downtime_hours"),
            self._association(merged, "BLASTING_DELAY", "blasting delay hours", "blasting_delay_hours"),
            self._association(merged, "RAINFALL", "rainfall", "rainfall_mm"),
        ]

        operations = self._operations_snapshot(merged, equipment_d, weather_d, blasting_d)
        signals = self._signals(merged, operations, associations)
        weighted = float(sum(signal["score"] for signal in signals) / max(len(signals), 1))
        overall_score = round(min(1.0, weighted), 3)

        coverage = {
            "production": int(production["date"].nunique()),
            "weather": int(weather_d["date"].isin(production["date"]).sum()),
            "equipment": int(equipment_d["date"].isin(production["date"]).sum()),
            "blasting": int(blasting_d["date"].isin(production["date"]).sum()),
            "aligned_days": int(len(merged)),
        }
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "latest_date": latest.date().isoformat(),
            "analysis_window_days": days,
            "production_records": int(len(production)),
            "production_mt_mean": round(float(production["production_mt"].mean()), 2),
            "target_mt_mean": round(float(production["target_mt"].mean()), 2),
            "gap_mt_mean": round(float(production["gap_mt"].mean()), 2),
            "fleet_availability": round(float(operations["fleet_availability"]), 3),
            "fleet_utilization": round(float(operations["fleet_utilization"]), 3),
            "rainfall_7d_mm": round(float(operations["rainfall_7d_mm"]), 2),
            "soil_moisture": round(float(operations["soil_moisture"]), 3),
            "planned_blasts_7d": int(operations["planned_blasts_7d"]),
            "blasting_delay_7d_hours": round(float(operations["blasting_delay_7d_hours"]), 2),
            "production_associations": associations,
            "risk_signals": signals,
            "overall_operational_risk": self._risk_level(overall_score),
            "overall_risk_score": overall_score,
            "data_coverage": coverage,
            "methodology_note": "Signals summarize observed same-day operational associations over the selected historical window. Correlation is not causation; no future values are used and this endpoint does not generate operational recommendations.",
        }

    @staticmethod
    def _operations_snapshot(merged: pd.DataFrame, equipment_d: pd.DataFrame, weather_d: pd.DataFrame, blasting_d: pd.DataFrame) -> dict[str, float | int]:
        latest = merged["date"].max()
        recent = merged[merged["date"] > latest - pd.Timedelta(days=7)]
        eq_recent = equipment_d[equipment_d["date"] > latest - pd.Timedelta(days=7)]
        weather_recent = weather_d[weather_d["date"] > latest - pd.Timedelta(days=7)]
        blast_recent = blasting_d[blasting_d["date"] > latest - pd.Timedelta(days=7)]
        utilization = float(eq_recent["utilization"].mean()) if not eq_recent.empty else 0.0
        downtime = float(eq_recent["downtime_hours"].sum()) if not eq_recent.empty else 0.0
        availability = max(0.0, 1.0 - min(1.0, downtime / (24.0 * max(eq_recent["date"].nunique(), 1) * max(1, 1)))) if not eq_recent.empty else 0.0
        return {
            "fleet_availability": availability,
            "fleet_utilization": utilization,
            "rainfall_7d_mm": float(weather_recent["rainfall_mm"].sum()) if not weather_recent.empty else 0.0,
            "soil_moisture": float(weather_recent["soil_moisture"].mean()) if not weather_recent.empty else 0.0,
            "planned_blasts_7d": int(blast_recent["planned_blasts"].sum()) if not blast_recent.empty else 0,
            "blasting_delay_7d_hours": float(blast_recent["blasting_delay_hours"].sum()) if not blast_recent.empty else 0.0,
            "recent_production_days": int(recent["date"].nunique()),
        }

    @classmethod
    def _signals(cls, merged: pd.DataFrame, snapshot: dict[str, float | int], associations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        downtime = float(snapshot["fleet_availability"])
        utilization = float(snapshot["fleet_utilization"])
        rainfall = float(snapshot["rainfall_7d_mm"])
        soil = float(snapshot["soil_moisture"])
        blast_delay = float(snapshot["blasting_delay_7d_hours"])
        blast_count = int(snapshot["planned_blasts_7d"])
        equipment_score = min(1.0, 0.65 * (1 - downtime) + 0.35 * max(0.0, min(1.0, (0.65 - utilization) / 0.65)))
        weather_score = min(1.0, 0.65 * min(1.0, rainfall / 90.0) + 0.35 * min(1.0, soil / 0.55))
        blast_score = min(1.0, 0.7 * min(1.0, blast_delay / 12.0) + 0.3 * min(1.0, blast_count / 5.0))
        corr_map = {item["driver"]: item for item in associations}
        cross = min(1.0, 0.4 * equipment_score + 0.3 * weather_score + 0.3 * blast_score)
        return [
            {"source": "EQUIPMENT", "level": cls._risk_level(equipment_score), "score": round(equipment_score, 3), "title": "Fleet availability and utilization signal", "evidence": {"fleet_availability": round(downtime, 3), "fleet_utilization": round(utilization, 3), "production_association": corr_map["EQUIPMENT_DOWNTIME"]["correlation"]}},
            {"source": "WEATHER", "level": cls._risk_level(weather_score), "score": round(weather_score, 3), "title": "Weather exposure signal", "evidence": {"rainfall_7d_mm": round(rainfall, 2), "soil_moisture": round(soil, 3), "production_association": corr_map["RAINFALL"]["correlation"]}},
            {"source": "BLASTING", "level": cls._risk_level(blast_score), "score": round(blast_score, 3), "title": "Blasting schedule and delay signal", "evidence": {"planned_blasts_7d": blast_count, "delay_hours_7d": round(blast_delay, 2), "production_association": corr_map["BLASTING_DELAY"]["correlation"]}},
            {"source": "CROSS_DOMAIN", "level": cls._risk_level(cross), "score": round(cross, 3), "title": "Combined operational exposure", "evidence": {"components": {"equipment": round(equipment_score, 3), "weather": round(weather_score, 3), "blasting": round(blast_score, 3)}, "method": "weighted risk-signal aggregation"}},
        ]
