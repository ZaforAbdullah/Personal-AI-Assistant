"""Events the agent emits while streaming a reply.

This is the wire contract with the frontend: each one becomes an SSE event named by its `type`,
mirrored in src/lib/types.ts.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter


class ConversationEvent(BaseModel):
    type: Literal["conversation"] = "conversation"
    conversation_id: UUID


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    call_id: str
    name: str
    args: dict[str, Any]


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    result: Any


class ToolErrorEvent(BaseModel):
    type: Literal["tool_error"] = "tool_error"
    call_id: str
    name: str
    error: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    message_id: UUID | None = None
    elapsed_ms: int | None = None  # wall time for the whole turn, model calls and tools included
    tool_calls: int = 0


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


AgentEvent = Annotated[
    ConversationEvent
    | TextDeltaEvent
    | ToolStartEvent
    | ToolResultEvent
    | ToolErrorEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


def openapi_event_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    """The event union as OpenAPI pieces: (the `oneOf` schema for the SSE body, the
    per-event component schemas it references). Refs point at `#/components/schemas/…`
    so Swagger UI can resolve them once the components are merged into the document."""
    schema = TypeAdapter(AgentEvent).json_schema(ref_template="#/components/schemas/{model}")
    components = schema.pop("$defs", {})
    return schema, components
