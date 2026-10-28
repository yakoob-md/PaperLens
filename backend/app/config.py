
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    google_api_key: str = ""
    adk_model: str = "gemini-2.0-flash"

    mcp_enabled: bool = True

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = 12
    max_text_chars: int = 80000

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
