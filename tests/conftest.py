from __future__ import annotations

import os
from pathlib import Path

import pytest

# Use a project-local temp directory to avoid platform-specific permission
# issues with the system temp (e.g. Windows "Access is denied" on the
# default pytest temp root). Must be set before pytest creates temp dirs.
_project_tmp = Path(__file__).resolve().parents[1] / ".pytest-tmp"
_project_tmp.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(_project_tmp))
os.environ.setdefault("TEMP", str(_project_tmp))
os.environ.setdefault("TMP", str(_project_tmp))

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
