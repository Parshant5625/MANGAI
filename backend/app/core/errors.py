from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


class MANGAIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelUnavailableError(MANGAIError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__("MODEL_UNAVAILABLE", message, status_code=503, details=details)


class DataUnavailableError(MANGAIError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__("DATA_UNAVAILABLE", message, status_code=503, details=details)


class NotFoundError(MANGAIError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, status_code=404, details=details)


class ValidationFailedError(MANGAIError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, status_code=422, details=details)


def error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(code, message, details))
