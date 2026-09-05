from __future__ import annotations

import argparse
import os

from alembic.config import Config

from alembic import command
from backend.app.core.config import Settings, get_settings
from backend.app.db import models as _models  # noqa: F401
from backend.app.db.session import SessionLocal
from backend.app.repositories.seed import seed_demo_database
from backend.app.services.demo_data import DemoDataStore


def run_migrations(settings: Settings) -> None:
    alembic_cfg = Config(str(settings.project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(settings.project_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(alembic_cfg, "head")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed MANGAI demo data and optional models")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.compact:
        os.environ["MANGAI_COMPACT_DATA"] = "1"
    settings = get_settings()
    store = DemoDataStore()
    store.ensure_demo_data()
    run_migrations(settings)
    with SessionLocal() as session:
        site_id = seed_demo_database(session, store=store)
    print(f"Seeded demo site {site_id} into {settings.database_url}")
    if not args.skip_train:
        from scripts.train_all import main as train_all

        train_all()


if __name__ == "__main__":
    main()
