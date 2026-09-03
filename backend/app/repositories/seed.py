from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.models import (
    BlastingEvent,
    BoreholeInterval,
    Equipment,
    EquipmentEvent,
    GeologicalSample,
    MineSite,
    ProductionRecord,
    SatelliteObservation,
    WeatherObservation,
    uuid_text,
)
from backend.app.services.demo_data import DemoDataStore


def seed_demo_database(session: Session, store: DemoDataStore | None = None, limit: int | None = 1500) -> str:
    settings = get_settings()
    store = store or DemoDataStore()
    store.ensure_demo_data()
    site = session.query(MineSite).filter_by(code=settings.demo_site_id).one_or_none()
    if site is None:
        site = MineSite(
            id=uuid_text(),
            name=settings.demo_site_name,
            code=settings.demo_site_id,
            latitude=21.4,
            longitude=80.3,
            area_geojson={"type": "Polygon", "coordinates": [[[80.1, 21.2], [80.5, 21.2], [80.5, 21.6], [80.1, 21.6], [80.1, 21.2]]]},
        )
        session.add(site)
        session.flush()

    if session.query(GeologicalSample).count() == 0:
        geo = store.geological().head(limit)
        sat = store.satellite().head(limit)
        for _, row in geo.iterrows():
            session.add(
                GeologicalSample(
                    site_id=site.id,
                    sample_id=str(row["sample_id"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    depth_m=float(row["depth_m"]),
                    elevation_m=float(row["elevation_m"]),
                    slope_deg=float(row["slope_deg"]),
                    aspect_deg=float(row["aspect_deg"]),
                    formation=str(row["formation"]),
                    mn_pct=float(row["mn_pct"]),
                    fe_pct=float(row["fe_pct"]),
                    sio2_pct=float(row["sio2_pct"]),
                    ore_thickness_m=float(row["ore_thickness_m"]),
                    source="demo_synthetic",
                )
            )
        for _, row in sat.iterrows():
            session.add(
                SatelliteObservation(
                    site_id=site.id,
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    blue_b2=float(row["blue_b2"]),
                    green_b3=float(row["green_b3"]),
                    red_b4=float(row["red_b4"]),
                    nir_b8=float(row["nir_b8"]),
                    swir_b11=float(row["swir_b11"]),
                    swir_b12=float(row["swir_b12"]),
                    ndvi=float(row["ndvi"]),
                    ndwi=float(row["ndwi"]),
                    swir_ratio=float(row["swir_ratio"]),
                    bare_soil_index=float(row["bare_soil_index"]),
                    lst_c=float(row.get("land_surface_temperature", 31)),
                    source="demo_synthetic",
                )
            )

    if session.query(BoreholeInterval).count() == 0:
        for _, row in store.boreholes().head(limit or 800).iterrows():
            session.add(
                BoreholeInterval(
                    site_id=site.id,
                    borehole_id=str(row["borehole_id"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    from_depth_m=float(row["from_depth_m"]),
                    to_depth_m=float(row["to_depth_m"]),
                    lithology=str(row["lithology"]),
                    mn_pct=float(row["mn_pct"]),
                    fe_pct=float(row["fe_pct"]),
                    sio2_pct=float(row["sio2_pct"]),
                    source="demo_synthetic",
                )
            )

    if session.query(WeatherObservation).count() == 0:
        weather = store.weather().tail(limit or 365)
        for _, row in weather.iterrows():
            observed = pd.Timestamp(row["date"]).to_pydatetime().replace(tzinfo=UTC)
            session.add(
                WeatherObservation(
                    site_id=site.id,
                    observed_at=observed,
                    rainfall_mm=float(row["rainfall_mm"]),
                    soil_moisture=float(row["soil_moisture"]),
                    temperature_c=float(row["temperature_c"]),
                    vegetation_index=float(row["vegetation_index"]),
                    source="demo_synthetic",
                )
            )

    if session.query(Equipment).count() == 0:
        equipment_df = store.equipment()
        latest = equipment_df.sort_values("date").groupby("equipment_id").tail(1)
        equipment_ids: dict[str, str] = {}
        for _, row in latest.iterrows():
            record = Equipment(
                site_id=site.id,
                equipment_id=str(row["equipment_id"]),
                equipment_type=str(row["equipment_type"]),
                capacity_tph=float(row["capacity_tph"]),
                active=True,
            )
            session.add(record)
            session.flush()
            equipment_ids[record.equipment_id] = record.id
        events = equipment_df.tail(limit or 2000)
        for _, row in events.iterrows():
            session.add(
                EquipmentEvent(
                    equipment_id=equipment_ids[str(row["equipment_id"])],
                    event_date=pd.Timestamp(row["date"]).to_pydatetime().replace(tzinfo=UTC),
                    operating_hours=float(row["operating_hours"]),
                    downtime_hours=float(row["downtime_hours"]),
                    utilization=float(row["utilization"]),
                    maintenance=bool(row["maintenance"]),
                )
            )

    if session.query(BlastingEvent).count() == 0:
        for _, row in store.blasting().tail(limit or 365).iterrows():
            session.add(
                BlastingEvent(
                    site_id=site.id,
                    event_date=pd.Timestamp(row["date"]).to_pydatetime().replace(tzinfo=UTC),
                    planned_blasts=int(row["planned_blasts"]),
                    delay_hours=float(row["blasting_delay_hours"]),
                    delay_reason=str(row["delay_reason"]),
                )
            )

    if session.query(ProductionRecord).count() == 0:
        for _, row in store.production().tail(limit or 365).iterrows():
            session.add(
                ProductionRecord(
                    site_id=site.id,
                    production_date=pd.Timestamp(row["date"]).date(),
                    actual_mt=float(row["production_mt"]),
                    target_mt=float(row["target_mt"]),
                    gap_mt=float(row["production_gap_mt"]),
                )
            )

    session.commit()
    return site.id
