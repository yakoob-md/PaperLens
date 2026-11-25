
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_input_usd_per_million: float | None = None
    gemini_output_usd_per_million: float | None = None

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_agent_model: str = ""
    groq_prompt_bytes: int = 5000
    groq_output_tokens: int = 2700
    groq_timeout_seconds: float = 60.0
    groq_max_retry_wait_seconds: float = 65.0
    groq_input_usd_per_million: float | None = None
    groq_output_usd_per_million: float | None = None
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    cloudflare_model: str = "@cf/openai/gpt-oss-20b"

    @property
    def cloudflare_configured(self) -> bool:
        return bool(self.cloudflare_account_id.strip() and self.cloudflare_api_token.strip())

    @property
    def groq_specialist_model(self) -> str:
        return self.groq_agent_model or self.groq_model

    mcp_enabled: bool = True
    a2a_critic_url: str = ""
    a2a_public_url: str = ""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = ""

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = 12

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
