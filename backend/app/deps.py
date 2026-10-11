"""FastAPI dependencies. Everything is built once at startup and stored on `app.state`."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.agent.service import ChatService
from app.cache import TTLCache
from app.config import Settings
from app.db.repositories import ConversationRepository, KnowledgeRepository
from app.llm.embeddings import Embedder


@dataclass(slots=True)
class AppState:
    settings: Settings
    model: BaseChatModel
    embedder: Embedder
    tools: list[BaseTool]
    conversations: ConversationRepository
    knowledge: KnowledgeRepository
    cache: TTLCache | None
    chat_service: ChatService = field(init=False)


def _state(request: Request) -> AppState:
    return request.app.state.container


def get_settings_dep(request: Request) -> Settings:
    return _state(request).settings


def get_conversation_repo(request: Request) -> ConversationRepository:
    return _state(request).conversations


def get_chat_service(request: Request) -> ChatService:
    return _state(request).chat_service


def get_cache(request: Request) -> TTLCache | None:
    return _state(request).cache


def get_tools(request: Request) -> list[BaseTool]:
    return _state(request).tools
