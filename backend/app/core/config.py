from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DataMode = Literal["demo", "live"]
SUPPORTED_DATA_MODES = {"demo", "live"}
SUPPORTED_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


class Settings(BaseSettings):
    app_name: str = "MANGAI"
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    api_v1_prefix: str = "/api/v1"
    data_mode: DataMode = Field(default="demo", validation_alias="DATA_MODE")
    database_url: str = Field(
        default="sqlite:///./mangai_dev.db",
        validation_alias="DATABASE_URL",
    )
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="CORS_ORIGINS",
    )
    model_dir: Path = Field(default=Path("models"), validation_alias="MODEL_DIR")
    data_dir: Path = Field(default=Path("data"), validation_alias="DATA_DIR")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    demo_site_id: str = "demo-moil-site"
    demo_site_name: str = "MANGAI Demo Mine"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_env(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("data_mode", mode="before")
    @classmethod
    def validate_data_mode(cls, value: object) -> str:
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_DATA_MODES:
            supported = ", ".join(sorted(SUPPORTED_DATA_MODES))
            raise ValueError(f"DATA_MODE must be one of: {supported}")
        return normalized

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: object) -> str:
        normalized = str(value).strip().upper()
        if normalized not in SUPPORTED_LOG_LEVELS:
            supported = ", ".join(sorted(SUPPORTED_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {supported}")
        return normalized

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else self.project_root / self.data_dir

    @property
    def resolved_model_dir(self) -> Path:
        return self.model_dir if self.model_dir.is_absolute() else self.project_root / self.model_dir

    @property
    def require_model_artifacts(self) -> bool:
        return self.data_mode == "live"


@lru_cache
def get_settings() -> Settings:
    return Settings()
