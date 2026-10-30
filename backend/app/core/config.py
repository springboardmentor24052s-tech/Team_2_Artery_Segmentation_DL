import os
from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Robust discovery of .env file (project root is parent of backend/ or current dir)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if not ENV_PATH.exists():
    # Fallback to local cwd .env
    ENV_PATH = Path(".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://postgres:8727@localhost:5432/artery_db",
        description="SQLAlchemy database connection string using psycopg 3 driver",
    )

    # Security & Tokens
    JWT_SECRET_KEY: str = Field(
        default="dev_secret_key_artery_stenosis_segmentation_auth_token_key_change_me",
        description="Secret key for signing JWT tokens",
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)

    # Environment
    ENVIRONMENT: str = Field(default="development")

    # CORS
    ALLOWED_ORIGINS: Union[List[str], str] = Field(default=["*"])

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


settings = Settings()

