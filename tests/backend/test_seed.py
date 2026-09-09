"""Phase 2 tests: database seeding idempotency and referential consistency."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db import models as _models  # noqa: F401 (register tables)
from backend.app.db.base import Base
from backend.app.repositories.seed import seed_demo_database


def _make_sqlite_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, session_factory()


def _counts(session) -> dict[str, int]:
    rows = session.execute(
        text(
            "SELECT 'mine_sites' AS t, count(*) FROM mine_sites "
            "UNION ALL SELECT 'geological_samples', count(*) FROM geological_samples "
            "UNION ALL SELECT 'boreholes', count(*) FROM boreholes "
            "UNION ALL SELECT 'satellite_observations', count(*) FROM satellite_observations "
            "UNION ALL SELECT 'weather_observations', count(*) FROM weather_observations "
            "UNION ALL SELECT 'equipment', count(*) FROM equipment "
            "UNION ALL SELECT 'equipment_events', count(*) FROM equipment_events "
            "UNION ALL SELECT 'blasting_events', count(*) FROM blasting_events "
            "UNION ALL SELECT 'production_records', count(*) FROM production_records"
        )
    ).fetchall()
    return {table: int(count) for table, count in rows}


def test_seed_is_idempotent(demo_store):
    """Running seed twice on the same database must not create duplicates."""
    engine, session = _make_sqlite_session()
    try:
        first = seed_demo_database(session, store=demo_store)
        counts_after_first = _counts(session)

        second = seed_demo_database(session, store=demo_store)
        counts_after_second = _counts(session)

        assert first == second, "site id must be stable across re-seeds"
        assert counts_after_first == counts_after_second, "second seed created duplicates"
        assert counts_after_first["mine_sites"] == 1
        assert counts_after_first["geological_samples"] > 0
        assert counts_after_first["production_records"] > 0
    finally:
        session.close()
        engine.dispose()


def test_seeded_data_references_the_demo_site(demo_store):
    from backend.app.core.config import get_settings

    engine, session = _make_sqlite_session()
    try:
        seed_demo_database(session, store=demo_store)
        site_row = session.execute(
            text("SELECT id, code FROM mine_sites WHERE code = :code"),
            {"code": get_settings().demo_site_id},
        ).fetchone()
        assert site_row is not None

        # Every domain record must reference the seeded site.
        for table, column in (
            ("geological_samples", "site_id"),
            ("satellite_observations", "site_id"),
            ("weather_observations", "site_id"),
            ("production_records", "site_id"),
            ("blasting_events", "site_id"),
            ("equipment", "site_id"),
        ):
            total = session.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            linked = session.execute(
                text(f"SELECT count(*) FROM {table} WHERE {column} = :sid"),
                {"sid": site_row.id},
            ).scalar()
            assert total == linked, f"{table} has rows that do not reference the demo site"
    finally:
        session.close()
        engine.dispose()


def test_equipment_events_reference_registered_equipment(demo_store):
    engine, session = _make_sqlite_session()
    try:
        seed_demo_database(session, store=demo_store)
        orphaned = session.execute(
            text(
                "SELECT count(*) FROM equipment_events e "
                "LEFT JOIN equipment q ON e.equipment_id = q.id "
                "WHERE q.id IS NULL"
            )
        ).scalar()
        assert orphaned == 0
    finally:
        session.close()
        engine.dispose()