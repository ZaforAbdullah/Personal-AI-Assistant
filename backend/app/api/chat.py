from __future__ import annotations

from contextlib import aclosing
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.events import openapi_event_schema
from app.agent.service import ChatService, ConversationNotFound
from app.config import Settings
from app.db.repositories import (
    TITLE_MAX_CHARS,
    ConversationRepository,
    ConversationSummary,
    MessageRecord,
)
from app.deps import get_chat_service, get_conversation_repo, get_settings_dep
from app.sse import sse_stream

router = APIRouter(prefix="/api", tags=["chat"])

# SSE payload schema for /docs; main.py merges EVENT_COMPONENTS into components/schemas
_EVENT_SCHEMA, EVENT_COMPONENTS = openapi_event_schema()
_SSE_EXAMPLE = (
    'event: conversation\ndata: {"type":"conversation","conversation_id":"…"}\n\n'
    'event: tool_start\ndata: {"type":"tool_start","call_id":"c1","name":"calculator",'
    '"args":{"expression":"2^10"}}\n\n'
    'event: tool_result\ndata: {"type":"tool_result","call_id":"c1","name":"calculator",'
    '"result":{"result":1024}}\n\n'
    'event: text_delta\ndata: {"type":"text_delta","text":"2^10 is 1024."}\n\n'
    'event: done\ndata: {"type":"done","message_id":"…","elapsed_ms":1830,"tool_calls":1}\n\n'
)

ServiceDep = Annotated[ChatService, Depends(get_chat_service)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
RepoDep = Annotated[ConversationRepository, Depends(get_conversation_repo)]


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=4000)
    # BCP-47 tag from navigator.language; the reply is written in it (English if missing)
    locale: str | None = Field(default=None, max_length=35, pattern=r"^[A-Za-z0-9_-]+$")


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    status: str
    created_at: str

    @classmethod
    def from_record(cls, m: MessageRecord) -> MessageOut:
        return cls(
            id=m.id,
            role=m.role,
            content=m.content,
            tool_calls=m.tool_calls,
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
            status=m.status,
            created_at=m.created_at.isoformat(),
        )


@router.post(
    "/chat",
    summary="Send a message and stream the reply",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "Server-Sent Events. One `conversation` event first, then any number of "
                "`text_delta`, `tool_start`, `tool_result` and `tool_error` events, then "
                "`done` (or `error` if the model failed; partial text is still stored). "
                "A `: ping` comment line is sent after 15 s of silence."
            ),
            "content": {"text/event-stream": {"schema": _EVENT_SCHEMA, "example": _SSE_EXAMPLE}},
        },
        404: {"description": "`conversation_id` does not exist"},
    },
)
async def chat(body: ChatRequest, service: ServiceDep, settings: SettingsDep) -> StreamingResponse:
    """Omit `conversation_id` to start a new conversation; its id arrives in the first
    event. `locale` (the browser's `navigator.language`) sets the reply language."""
    events = service.stream(body.conversation_id, body.message, locale=body.locale)
    # pull the first event now so a missing conversation is a 404, not an error frame in a 200
    try:
        first = await anext(events)
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc

    async def with_first():
        async with aclosing(events):  # closes the service stream if the client leaves early
            yield first
            async for event in events:
                yield event

    return StreamingResponse(
        sse_stream(with_first(), heartbeat_s=settings.sse_heartbeat_s),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ConversationOut(BaseModel):
    id: UUID
    title: str
    created_at: str
    updated_at: str
    message_count: int

    @classmethod
    def from_summary(cls, s: ConversationSummary) -> ConversationOut:
        return cls(
            id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
            message_count=s.message_count,
        )


@router.get("/conversations", summary="List conversations for the sidebar")
async def list_conversations(
    repo: RepoDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[ConversationOut]:
    """Most recently active first. Only conversations with at least one message."""
    return [ConversationOut.from_summary(s) for s in await repo.list_conversations(limit=limit)]


@router.get(
    "/conversations/{conversation_id}/messages",
    summary="Full stored history of a conversation",
    responses={404: {"description": "Unknown conversation"}},
)
async def list_messages(conversation_id: UUID, repo: RepoDep) -> list[MessageOut]:
    """Every row in order: `user`, `assistant` (with `tool_calls`), and `tool` results.
    `status` is `complete`, `error` (model failed mid-reply) or `interrupted` (client
    stopped it); the partial text is kept in both cases."""
    if await repo.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return [MessageOut.from_record(m) for m in await repo.list_messages(conversation_id)]


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    summary="Delete a conversation and its messages",
    responses={204: {"description": "Deleted"}, 404: {"description": "Unknown conversation"}},
)
async def delete_conversation(conversation_id: UUID, repo: RepoDep) -> None:
    if not await repo.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=TITLE_MAX_CHARS)

    @field_validator("title")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("title must not be blank")
        return cleaned


@router.patch(
    "/conversations/{conversation_id}",
    summary="Rename a conversation",
    responses={404: {"description": "Unknown conversation"}},
)
async def rename_conversation(
    conversation_id: UUID, body: RenameRequest, repo: RepoDep
) -> ConversationOut:
    if not await repo.rename_conversation(conversation_id, body.title):
        raise HTTPException(status_code=404, detail="conversation not found")
    summary = next(
        (s for s in await repo.list_conversations(limit=200) if s.id == conversation_id), None
    )
    if summary is None:  # renamed an empty conversation (no messages yet)
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationOut.from_summary(summary)
