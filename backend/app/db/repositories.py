"""Storage contracts and in-memory implementations.

The service depends only on the Protocols here. Postgres lives in `app.db.postgres`; the
in-memory versions back the offline tests.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

MessageStatus = Literal["complete", "error", "interrupted"]


@dataclass(slots=True)
class ConversationRecord:
    id: UUID
    created_at: datetime
    title: str | None = None  # user-chosen; None means "use the first user message"


@dataclass(slots=True)
class ConversationSummary:
    """What a conversation list needs: a title (custom, else the first user message) and
    recency."""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


TITLE_MAX_CHARS = 80


def make_title(first_user_message: str) -> str:
    text = " ".join(first_user_message.split())
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1] + "…"


@dataclass(slots=True)
class MessageRecord:
    conversation_id: UUID
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None  # assistant: [{id, name, arguments}]
    tool_call_id: str | None = None
    tool_name: str | None = None
    status: MessageStatus = "complete"
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class KnowledgeChunk:
    id: UUID
    source: str
    content: str
    embedding: list[float]
    embedding_model: str


@dataclass(slots=True)
class KnowledgeHit:
    source: str
    content: str
    score: float  # cosine similarity, higher is better


class ConversationRepository(Protocol):
    async def create_conversation(self) -> ConversationRecord: ...

    async def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None: ...

    async def add_messages(self, messages: Iterable[MessageRecord]) -> None: ...

    async def list_messages(
        self, conversation_id: UUID, *, last: int | None = None
    ) -> list[MessageRecord]:
        """Messages in chronological order; `last` keeps only the most recent N."""
        ...

    async def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        """Conversations that have at least one message, most recently active first."""
        ...

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete a conversation and its messages. Returns False if it did not exist."""
        ...

    async def rename_conversation(self, conversation_id: UUID, title: str) -> bool:
        """Set a user-chosen title. Returns False if the conversation does not exist."""
        ...

    async def ping(self) -> bool: ...


# What the knowledge base should hold: (source, content) pairs, order irrelevant.
WantedChunks = Sequence[tuple[str, str]]
EmbedTexts = Callable[[list[str]], Awaitable[list[list[float]]]]


class KnowledgeRepository(Protocol):
    async def count(self, embedding_model: str) -> int: ...

    async def replace_all(self, chunks: Iterable[KnowledgeChunk]) -> None: ...

    async def sync(self, embedding_model: str, wanted: WantedChunks, embed: EmbedTexts) -> int:
        """Make the store hold exactly `wanted`, embedded with `embedding_model`.

        A no-op (and `embed` is never called) when it already does, all-or-nothing if `embed` fails,
        and racing callers are serialised. Returns the number of chunks embedded.
        """
        ...

    async def search(self, embedding: list[float], *, limit: int) -> list[KnowledgeHit]: ...


class InMemoryConversationRepository:
    def __init__(self) -> None:
        self._conversations: dict[UUID, ConversationRecord] = {}
        self._messages: dict[UUID, list[MessageRecord]] = {}

    async def create_conversation(self) -> ConversationRecord:
        conv = ConversationRecord(id=uuid4(), created_at=datetime.now(UTC))
        self._conversations[conv.id] = conv
        self._messages[conv.id] = []
        return conv

    async def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None:
        return self._conversations.get(conversation_id)

    async def add_messages(self, messages: Iterable[MessageRecord]) -> None:
        for m in messages:
            self._messages.setdefault(m.conversation_id, []).append(m)

    async def list_messages(
        self, conversation_id: UUID, *, last: int | None = None
    ) -> list[MessageRecord]:
        msgs = list(self._messages.get(conversation_id, []))
        return msgs[-last:] if last else msgs

    async def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        summaries: list[ConversationSummary] = []
        for conv in self._conversations.values():
            msgs = self._messages.get(conv.id, [])
            if not msgs:
                continue
            first_user = next((m.content for m in msgs if m.role == "user"), "")
            summaries.append(
                ConversationSummary(
                    id=conv.id,
                    title=conv.title or make_title(first_user) or "New chat",
                    created_at=conv.created_at,
                    updated_at=max(m.created_at for m in msgs),
                    message_count=len(msgs),
                )
            )
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        existed = self._conversations.pop(conversation_id, None) is not None
        self._messages.pop(conversation_id, None)
        return existed

    async def rename_conversation(self, conversation_id: UUID, title: str) -> bool:
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return False
        conv.title = title
        return True

    async def ping(self) -> bool:
        return True


class InMemoryKnowledgeRepository:
    def __init__(self) -> None:
        self._chunks: list[KnowledgeChunk] = []
        self._sync_lock = asyncio.Lock()

    async def sync(self, embedding_model: str, wanted: WantedChunks, embed: EmbedTexts) -> int:
        async with self._sync_lock:
            have = Counter((c.embedding_model, c.source, c.content) for c in self._chunks)
            if have == Counter((embedding_model, s, c) for s, c in wanted):
                return 0
            vectors = await embed([c for _, c in wanted])  # may raise: nothing changed yet
            self._chunks = _build_chunks(embedding_model, wanted, vectors)
            return len(self._chunks)

    async def count(self, embedding_model: str) -> int:
        return sum(1 for c in self._chunks if c.embedding_model == embedding_model)

    async def replace_all(self, chunks: Iterable[KnowledgeChunk]) -> None:
        self._chunks = list(chunks)

    async def search(self, embedding: list[float], *, limit: int) -> list[KnowledgeHit]:
        scored = [
            KnowledgeHit(source=c.source, content=c.content, score=_cosine(embedding, c.embedding))
            for c in self._chunks
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]


def _build_chunks(
    embedding_model: str, wanted: WantedChunks, vectors: list[list[float]]
) -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            id=uuid4(), source=s, content=c, embedding=v, embedding_model=embedding_model
        )
        for (s, c), v in zip(wanted, vectors, strict=True)
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
