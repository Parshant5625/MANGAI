from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.adapters.factory import weather_provider
from backend.app.core.config import get_settings
from backend.app.services.demo_data import DemoDataStore, demo_envelope


class OperationsService:
    def __init__(self, store: DemoDataStore | None = None) -> None:
        self.store = store or DemoDataStore()
        self.settings = get_settings()

    def equipment(self, site_id: str | None = None) -> dict[str, Any]:
        df = self.store.equipment().copy()
        latest = pd.Timestamp(df["date"].max())
        last_7 = df[df["date"] > latest - pd.Timedelta(days=7)]
        last_30 = df[df["date"] > latest - pd.Timedelta(days=30)]
        rows = []
        for equipment_id, group in last_30.groupby("equipment_id"):
            group7 = last_7[last_7["equipment_id"] == equipment_id]
            availability = 1 - min(1.0, float(group7["downtime_hours"].sum()) / (24 * max(len(group7), 1)))
            utilization = float(group7["utilization"].mean())
            downtime_7 = float(group7["downtime_hours"].sum())
            downtime_30 = float(group["downtime_hours"].sum())
            maintenance_30 = int(group["maintenance"].sum())
            status = "NORMAL"
            if downtime_7 >= 18 or utilization < 0.5:
                status = "CRITICAL"
            elif downtime_7 >= 10 or utilization < 0.58 or maintenance_30 >= 3:
                status = "WATCH"
            rows.append(
                {
                    "equipment_id": str(equipment_id),
                    "equipment_type": str(group["equipment_type"].iloc[-1]),
                    "availability": round(float(availability), 3),
                    "utilization": round(float(utilization), 3),
                    "downtime_7d_hours": round(downtime_7, 2),
                    "downtime_30d_hours": round(downtime_30, 2),
                    "maintenance_events_30d": maintenance_30,
                    "status": status,
                }
            )
        rows = sorted(rows, key=lambda item: item["downtime_7d_hours"], reverse=True)
        trend = (
            last_30.groupby(last_30["date"].dt.date)["maintenance"]
            .sum()
            .reset_index()
            .rename(columns={"date": "date", "maintenance": "maintenance_events"})
        )
        trend["date"] = trend["date"].astype(str)
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "fleet_availability": round(float(sum(item["availability"] for item in rows) / max(len(rows), 1)), 3),
            "fleet_utilization": round(float(sum(item["utilization"] for item in rows) / max(len(rows), 1)), 3),
            "critical_equipment_count": sum(1 for item in rows if item["status"] == "CRITICAL"),
            "items": rows,
            "maintenance_trend": trend.round(2).to_dict(orient="records"),
        }

    def weather(self, site_id: str | None = None) -> dict[str, Any]:
        records = weather_provider().fetch_forecast(site_id or self.settings.demo_site_id, "", "")
        df = pd.DataFrame(records)
        if df.empty:
            df = self.store.weather().copy()
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.get("observed_at", pd.Timestamp.utcnow()))
        df["date"] = pd.to_datetime(df["date"])
        latest = pd.Timestamp(df["date"].max())
        last_7 = df[df["date"] > latest - pd.Timedelta(days=7)]
        last_30 = df[df["date"] > latest - pd.Timedelta(days=30)]
        rainfall_7 = float(last_7["rainfall_mm"].sum())
        rainfall_30 = float(last_30["rainfall_mm"].sum())
        soil_moisture = float(last_7["soil_moisture"].mean())
        if rainfall_7 > 90 or soil_moisture > 0.55:
            risk = "HIGH"
        elif rainfall_7 > 35 or soil_moisture > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        obs = df.tail(30).copy()
        obs["date"] = obs["date"].dt.date.astype(str)
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "latest_date": latest.date().isoformat(),
            "rainfall_7d_mm": round(rainfall_7, 2),
            "rainfall_30d_mm": round(rainfall_30, 2),
            "soil_moisture": round(soil_moisture, 3),
            "temperature_c": round(float(last_7["temperature_c"].mean()), 2),
            "weather_risk": risk,
            "observations": obs.round(3).to_dict(orient="records"),
        }

    def blasting(self, site_id: str | None = None) -> dict[str, Any]:
        df = self.store.blasting().copy()
        weather = self.weather(site_id)
        latest = pd.Timestamp(df["date"].max())
        last_7 = df[df["date"] > latest - pd.Timedelta(days=7)]
        prev_7 = df[(df["date"] <= latest - pd.Timedelta(days=7)) & (df["date"] > latest - pd.Timedelta(days=14))]
        delay_7 = float(last_7["blasting_delay_hours"].sum())
        prev_delay = float(prev_7["blasting_delay_hours"].sum())
        if delay_7 > prev_delay * 1.15:
            trend = "WORSENING"
        elif delay_7 < prev_delay * 0.85:
            trend = "IMPROVING"
        else:
            trend = "STABLE"
        planned = int(last_7["planned_blasts"].sum())
        overlap = "LOW"
        if weather["weather_risk"] == "HIGH" and planned > 0:
            overlap = "HIGH"
        elif weather["weather_risk"] == "MEDIUM" and planned > 2:
            overlap = "MEDIUM"
        events = df.tail(30).copy()
        events["date"] = events["date"].dt.date.astype(str)
        return {
            **demo_envelope(),
            "site_id": site_id or self.settings.demo_site_id,
            "latest_date": latest.date().isoformat(),
            "planned_blasts_7d": planned,
            "delay_hours_7d": round(delay_7, 2),
            "delay_trend": trend,
            "overlap_risk": overlap,
            "events": events.round(3).to_dict(orient="records"),
        }
