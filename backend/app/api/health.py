from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.cache import TTLCache
from app.config import Settings
from app.db.repositories import ConversationRepository
from app.deps import get_cache, get_conversation_repo, get_settings_dep, get_tools

router = APIRouter(tags=["health"])


class RootOut(BaseModel):
    service: str
    docs: str
    health: str


class CacheStatsOut(BaseModel):
    hits: int
    misses: int
    entries: int
    hit_rate: float = Field(description="hits / (hits + misses), 0 when nothing was asked yet")


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"] = Field(description="`degraded` when the database ping fails")
    llm_provider: str
    llm_model: str
    db_provider: str
    db_ok: bool
    cache: CacheStatsOut | None = Field(description="null when CACHE_ENABLED=false")
    tools: list[str] = Field(
        description="Tools currently offered to the model; gif_search only with GIPHY_API_KEY"
    )


@router.get("/", summary="Where things are", response_model=RootOut)
async def root() -> RootOut:
    """The API has no UI of its own (the chat UI is the frontend on port 3000); this
    points a browser hitting the bare backend URL somewhere useful instead of a 404."""
    return RootOut(service="personal-ai-assistant", docs="/docs", health="/health")


@router.api_route(
    "/health",
    methods=["GET", "HEAD"],
    summary="Liveness, provider, tools and cache stats",
    response_model=HealthOut,
)
async def health(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    repo: Annotated[ConversationRepository, Depends(get_conversation_repo)],
    cache: Annotated[TTLCache | None, Depends(get_cache)],
    tools: Annotated[list[BaseTool], Depends(get_tools)],
) -> HealthOut:
    """Cheap enough to poll. The frontend reads it once to show the provider in the
    sidebar and to decide which suggestion chips to offer."""
    try:
        db_ok = await repo.ping()
    except Exception:
        db_ok = False
    return HealthOut(
        status="ok" if db_ok else "degraded",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        db_provider=settings.db_provider,
        db_ok=db_ok,
        cache=CacheStatsOut(**cache.stats.as_dict()) if cache is not None else None,
        tools=[t.name for t in tools],
    )
