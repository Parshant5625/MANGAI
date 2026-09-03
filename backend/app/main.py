from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sqlalchemy import text

from backend.app.api.v1.router import router as api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.core.security import router as security_router
from backend.app.db.session import engine
from backend.app.services.demo_data import BOUNDARY_NOTICE


settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MANGAI API in %s mode", settings.data_mode)
    from backend.app.db.base import Base
    from backend.app.db import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    yield
    logger.info("Stopping MANGAI API")


app = FastAPI(
    title="MANGAI API",
    description="AI-powered manganese mining intelligence platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
app.include_router(security_router, prefix=settings.api_v1_prefix)


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "detail": str(exc),
            "hint": "Run python scripts/seed_demo.py to generate offline demo data.",
        },
    )


@app.get("/")
def root() -> dict[str, object]:
    return {
        "system": "MANGAI",
        "status": "online",
        "version": app.version,
        "api": settings.api_v1_prefix,
        "data_mode": settings.data_mode,
        "boundary_notice": BOUNDARY_NOTICE,
    }


def _database_ok() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "mangai-api"}


@app.get("/ready")
def ready() -> dict[str, object]:
    database_ok = _database_ok()
    model_dir = settings.resolved_model_dir
    prospectivity = (model_dir / "reserve" / "prospectivity_xgboost.json").exists() or (
        model_dir / "reserve_xgboost.json"
    ).exists()
    production_model = (model_dir / "production" / "forecast_xgboost.json").exists()
    status = "ready" if database_ok else "degraded"
    return {
        "status": status,
        "data_mode": settings.data_mode,
        "database": database_ok,
        "prospectivity_model": prospectivity,
        "production_model": production_model,
        "data_dir": str(settings.resolved_data_dir),
        "model_dir": str(model_dir),
    }

