"""Settings from environment variables and the repo-root .env.

Provider switches (Gemini or LM Studio, Postgres or Supabase) are resolved here, so the rest
of the code only sees llm_base_url, llm_model, db_url and so on.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"

LLMProvider = Literal["gemini", "lmstudio"]
DBProvider = Literal["postgres", "supabase"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_REPO_ROOT_ENV, extra="ignore")

    llm_provider: LLMProvider = "lmstudio"

    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"  # LM Studio ignores the key but the SDK requires one
    lmstudio_model: str = "google/gemma-4-e4b"
    lmstudio_embedding_model: str = "text-embedding-nomic-embed-text-v1.5"

    embedding_dim: int = 768

    # low temperature keeps tool arguments well-formed; the cap bounds reply latency
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    db_provider: DBProvider = "postgres"
    database_url: str = "postgresql+asyncpg://assistant:assistant@localhost:5432/assistant"
    supabase_db_url: str = ""

    max_tool_rounds: int = 4
    tool_timeout_s: float = 15.0
    # gif_search is only offered to the model when a GIPHY key is configured.
    giphy_api_key: str = ""
    history_limit: int = 30
    sse_heartbeat_s: float = 15.0

    cache_enabled: bool = True
    cache_ttl_s: float = 300.0
    cache_max_entries: int = 2000

    # any hostname, but only the frontend's port. There is no auth or cookie here, so the
    # hostname isn't worth restricting; the port is what keeps other local apps out.
    frontend_port: int = 3000
    # extra exact origins for a deployed frontend, comma separated (https://app.example.com)
    cors_origins: str = ""

    @property
    def llm_base_url(self) -> str:
        return self.gemini_base_url if self.llm_provider == "gemini" else self.lmstudio_base_url

    @property
    def llm_api_key(self) -> str:
        return self.gemini_api_key if self.llm_provider == "gemini" else self.lmstudio_api_key

    @property
    def llm_model(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.lmstudio_model

    @property
    def embedding_model(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_embedding_model
        return self.lmstudio_embedding_model

    @property
    def db_url(self) -> str:
        if self.db_provider == "supabase":
            if not self.supabase_db_url:
                raise ValueError("DB_PROVIDER=supabase requires SUPABASE_DB_URL")
            return asyncpg_url(self.supabase_db_url)
        return asyncpg_url(self.database_url)

    @property
    def db_connect_args(self) -> dict[str, object]:
        # Supabase requires TLS; asyncpg accepts ssl="require".
        return {"ssl": "require"} if self.db_provider == "supabase" else {}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def frontend_origin_regex(self) -> str:
        """CORS origin pattern: any http(s) host, exactly `frontend_port`, nothing else."""
        return rf"^https?://[^/]+:{self.frontend_port}$"


def asyncpg_url(url: str) -> str:
    """Accept a connection string as a provider shows it and return one asyncpg can use.

    Hosts hand out `postgres://`, `postgresql://` or a psycopg variant; this app drives
    Postgres through asyncpg, which also rejects the libpq `sslmode` parameter (TLS is
    configured separately).
    """
    url = re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql+asyncpg://", url.strip())
    return re.sub(r"([?&])sslmode=[^&]*&?", r"\1", url).rstrip("?&")


@lru_cache
def get_settings() -> Settings:
    return Settings()
