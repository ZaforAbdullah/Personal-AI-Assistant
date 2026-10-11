"""FastAPI application factory.

`create_app()` accepts pre-built collaborators so tests can inject a scripted chat model
and in-memory repositories; in production everything is built from `Settings`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.agent.service import ChatService
from app.api import chat, health
from app.cache import TTLCache
from app.config import Settings, get_settings
from app.db.cached import CachedConversationRepository
from app.db.postgres import (
    PostgresConversationRepository,
    PostgresKnowledgeRepository,
    create_engine,
    init_db,
)
from app.db.repositories import ConversationRepository, KnowledgeRepository
from app.db.seed import seed_knowledge
from app.deps import AppState
from app.llm.cached import CachedEmbedder
from app.llm.embeddings import Embedder
from app.llm.errors import LLMError
from app.llm.factory import build_chat_model, build_embedder, close_chat_model
from app.middleware import RequestContextMiddleware
from app.tools.calculator import CalculatorTool
from app.tools.gif import GifSearchTool
from app.tools.http import make_client
from app.tools.knowledge_search import KnowledgeSearchTool
from app.tools.weather import WeatherTool

log = logging.getLogger(__name__)


async def _aclose(resource: object) -> None:
    """Close a resource that has an `aclose()`; tolerate ones that do not."""
    closer = getattr(resource, "aclose", None)
    if closer is not None:
        await closer()


def _drop_state(app: FastAPI) -> None:
    """Release the references the app held for its whole life (graph, cache, clients)."""
    if hasattr(app.state, "container"):
        del app.state.container


def create_app(
    settings: Settings | None = None,
    *,
    chat_model: BaseChatModel | None = None,
    embedder: Embedder | None = None,
    conversations: ConversationRepository | None = None,
    knowledge: KnowledgeRepository | None = None,
    seed: bool = True,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """FastAPI lifespan: startup before `yield`, shutdown after.

        Each resource registers its cleanup on the exit stack as soon as it is acquired, so a
        failure half-way through startup still releases what was opened. Only resources created
        here are closed; injected ones belong to the caller.
        """
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

        async with AsyncExitStack() as stack:
            conv_repo, kb_repo = conversations, knowledge
            if conv_repo is None or kb_repo is None:
                engine = create_engine(settings)
                stack.push_async_callback(engine.dispose)  # closes the connection pool
                await init_db(engine)
                conv_repo = conv_repo or PostgresConversationRepository(engine)
                kb_repo = kb_repo or PostgresKnowledgeRepository(engine)

            emb = embedder
            if emb is None:
                emb = build_embedder(settings)
                stack.push_async_callback(_aclose, emb)

            cache: TTLCache | None = None
            if settings.cache_enabled:
                cache = TTLCache(max_entries=settings.cache_max_entries, ttl_s=settings.cache_ttl_s)
                stack.callback(cache.clear)
                conv_repo = CachedConversationRepository(conv_repo, cache)
                emb = CachedEmbedder(emb, cache)

            if seed:
                try:
                    await seed_knowledge(kb_repo, emb)
                except LLMError as exc:
                    # a down embedding provider shouldn't stop the app; knowledge_search will say so
                    log.warning("knowledge base not seeded (embedding provider down?): %s", exc)
                except Exception:
                    # same for anything unexpected: log the traceback, keep chat up
                    log.exception("knowledge base not seeded; knowledge_search will be unavailable")

            http = make_client()
            stack.push_async_callback(http.aclose)
            tools: list[BaseTool] = [
                CalculatorTool(),
                KnowledgeSearchTool(embedder=emb, repo=kb_repo),
                WeatherTool(client=http),
            ]
            if settings.giphy_api_key:
                tools.append(GifSearchTool(api_key=settings.giphy_api_key, client=http))
            else:
                log.info("gif_search not offered: GIPHY_API_KEY is empty")

            model = chat_model
            if model is None:
                model = build_chat_model(settings)
                stack.push_async_callback(close_chat_model, model)

            state = AppState(
                settings=settings,
                model=model,
                embedder=emb,
                tools=tools,
                conversations=conv_repo,
                knowledge=kb_repo,
                cache=cache,
            )
            state.chat_service = ChatService(
                repo=state.conversations,
                model=state.model,
                tools=state.tools,
                max_tool_rounds=settings.max_tool_rounds,
                tool_timeout_s=settings.tool_timeout_s,
                history_limit=settings.history_limit,
            )
            stack.push_async_callback(state.chat_service.aclose)  # drain turns, then close pool
            app.state.container = state
            stack.callback(_drop_state, app)  # like `ml_models.clear()` in the FastAPI docs
            log.info(
                "ready: llm=%s (%s) db=%s tools=%s cache=%s",
                settings.llm_provider,
                settings.llm_model,
                settings.db_provider,
                [t.name for t in state.tools],
                f"ttl={settings.cache_ttl_s:g}s max={settings.cache_max_entries}"
                if cache is not None
                else "off",
            )

            yield

            log.info("shutting down: draining turns, then releasing pool, clients and cache")

    app = FastAPI(
        title="Personal AI Assistant",
        version="0.1.0",
        summary="A streaming chat assistant that can use tools.",
        description=(
            "Send a message with `POST /api/chat` and read the reply as Server-Sent Events: "
            "text arrives token by token, and every tool call is announced (`tool_start`) "
            "and resolved (`tool_result` / `tool_error`) on the same stream. Conversations "
            "are stored, listed, renamable and deletable.\n\n"
            "Tools: `calculator`, `knowledge_search` (pgvector over a seeded knowledge base), "
            "`weather` (Open-Meteo) and, when a GIPHY key is configured, `gif_search`. "
            "`GET /health` lists which are live.\n\n"
            "No authentication: this is a demo. CORS admits any host on the "
            "frontend's port."
        ),
        openapi_tags=[
            {"name": "chat", "description": "Streaming chat and conversation management."},
            {"name": "health", "description": "Liveness and configuration at a glance."},
        ],
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        # the frontend's port on any hostname, plus any exact origins listed in CORS_ORIGINS
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.frontend_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)

    def openapi_with_event_components() -> dict:
        # FastAPI only registers models it sees in signatures, so add the event models by hand
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        schema.setdefault("components", {}).setdefault("schemas", {}).update(chat.EVENT_COMPONENTS)
        app.openapi_schema = schema
        return schema

    app.openapi = openapi_with_event_components  # type: ignore[method-assign]
    return app


app = create_app()
