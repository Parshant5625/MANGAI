from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.v1.router import router as api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.errors import MANGAIError, error_response
from backend.app.core.logging import configure_logging
from backend.app.core.security import router as security_router
from backend.app.db.session import dispose_engine, is_database_available
from backend.app.services.demo_data import BOUNDARY_NOTICE
from backend.app.services.model_artifacts import model_status

configure_logging(get_settings())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting MANGAI API app_env=%s data_mode=%s data_dir=%s model_dir=%s",
        settings.app_env,
        settings.data_mode,
        settings.resolved_data_dir,
        settings.resolved_model_dir,
    )
    try:
        yield
    finally:
        logger.info("Stopping MANGAI API")
        dispose_engine()


app = FastAPI(
    title="MANGAI API",
    description="AI-powered manganese mining intelligence platform",
    version="1.0.0",
    lifespan=lifespan,
)

initial_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=initial_settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=initial_settings.api_v1_prefix)
app.include_router(security_router, prefix=initial_settings.api_v1_prefix)


def _http_error_code(status_code: int) -> str:
    if status_code == 404:
        return "NOT_FOUND"
    if status_code == 422:
        return "VALIDATION_ERROR"
    if status_code == 401:
        return "UNAUTHORIZED"
    if status_code == 403:
        return "FORBIDDEN"
    return "HTTP_ERROR"


def _validation_details(exc: RequestValidationError) -> dict[str, Any]:
    return {
        "errors": [
            {
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg", "Invalid request."),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
    }


@app.exception_handler(MANGAIError)
async def mangai_error_handler(request: Request, exc: MANGAIError) -> JSONResponse:
    logger.warning("API error path=%s code=%s message=%s", request.url.path, exc.code, exc.message)
    return error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("Request validation failed path=%s errors=%s", request.url.path, len(exc.errors()))
    return error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="Request validation failed.",
        details=_validation_details(exc),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "HTTP request failed."
    logger.info("HTTP error path=%s status_code=%s detail=%s", request.url.path, exc.status_code, detail)
    return error_response(
        status_code=exc.status_code,
        code=_http_error_code(exc.status_code),
        message=detail,
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("Data loading failed path=%s error=%s", request.url.path, exc)
    return error_response(
        status_code=503,
        code="DATA_UNAVAILABLE",
        message="Required data is not available.",
        details={"hint": "Run python scripts/seed_demo.py --skip-train to generate offline demo data."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled endpoint failure path=%s", request.url.path)
    return error_response(
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred.",
    )


@app.get("/")
def root() -> dict[str, object]:
    settings = get_settings()
    return {
        "system": "MANGAI",
        "status": "online",
        "version": app.version,
        "api": settings.api_v1_prefix,
        "data_mode": settings.data_mode,
        "boundary_notice": BOUNDARY_NOTICE,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "mangai-api"}


@app.get("/ready")
def ready() -> dict[str, object]:
    settings = get_settings()
    database_ok = is_database_available()
    models = model_status(settings)
    demo_data_ok = True
    if settings.data_mode == "demo":
        from backend.app.services.demo_data import DemoDataStore

        demo_data_ok = DemoDataStore().demo_data_available()
    models_required = settings.data_mode == "live"
    dependencies_ok = database_ok and demo_data_ok and (not models_required or all(models.values()))
    status = "ready" if dependencies_ok else "degraded"
    model_dir = settings.resolved_model_dir
    return {
        "status": status,
        "database": database_ok,
        "data_mode": settings.data_mode,
        "models": models,
        "demo_data": demo_data_ok,
        "data_dir": str(settings.resolved_data_dir),
        "model_dir": str(model_dir),
    }
