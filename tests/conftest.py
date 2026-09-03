from __future__ import annotations

import os

import pytest

os.environ.setdefault("MANGAI_COMPACT_DATA", "1")
os.environ.setdefault("DATA_MODE", "demo")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="session")
def demo_store():
    from backend.app.core.config import get_settings
    from backend.app.services.demo_data import DemoDataStore

    get_settings.cache_clear()
    store = DemoDataStore()
    store.ensure_demo_data()
    return store
