"""Read-through cache in front of a ConversationRepository.

Message windows are extended in place on writes, so a conversation being talked to never
re-reads its history; lists and single records are invalidated on any write.

Keys: ("conv", id), ("msgs", id, last) with last=None for the whole history, ("list", limit).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import UUID

from app.cache import MISSING, TTLCache
from app.db.repositories import (
    ConversationRecord,
    ConversationRepository,
    ConversationSummary,
    MessageRecord,
)

Key = tuple[Any, ...]

# the full-history entry is unbounded, so past this many messages a conversation is read
# from the database instead of cached
DEFAULT_MAX_CACHED_MESSAGES = 200


class CachedConversationRepository:
    def __init__(
        self,
        inner: ConversationRepository,
        cache: TTLCache[Key, Any],
        *,
        max_cached_messages: int = DEFAULT_MAX_CACHED_MESSAGES,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._max_messages = max_cached_messages

    async def create_conversation(self) -> ConversationRecord:
        conv = await self._inner.create_conversation()
        self._cache.set(("conv", conv.id), conv)
        self._cache.set(("msgs", conv.id, None), [])
        self._invalidate_lists()
        return conv

    async def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None:
        key = ("conv", conversation_id)
        hit = self._cache.get(key)
        if hit is not MISSING:
            return hit
        conv = await self._inner.get_conversation(conversation_id)
        if conv is not None:
            self._cache.set(key, conv)
        return conv

    async def list_messages(
        self, conversation_id: UUID, *, last: int | None = None
    ) -> list[MessageRecord]:
        key = ("msgs", conversation_id, last)
        hit = self._cache.get(key)
        if hit is not MISSING:
            return list(hit)
        # A cached full history can answer any window without a query.
        full = self._cache.get(("msgs", conversation_id, None))
        if full is not MISSING:
            window = full[-last:] if last else list(full)
            self._cache.set(key, window)
            return list(window)
        msgs = await self._inner.list_messages(conversation_id, last=last)
        if len(msgs) <= self._max_messages:
            self._cache.set(key, list(msgs))
        return msgs

    async def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        key = ("list", limit)
        hit = self._cache.get(key)
        if hit is not MISSING:
            return list(hit)
        summaries = await self._inner.list_conversations(limit=limit)
        self._cache.set(key, list(summaries))
        return summaries

    async def ping(self) -> bool:
        return await self._inner.ping()

    async def add_messages(self, messages: Iterable[MessageRecord]) -> None:
        new = list(messages)
        await self._inner.add_messages(new)
        by_conv: dict[UUID, list[MessageRecord]] = {}
        for m in new:
            by_conv.setdefault(m.conversation_id, []).append(m)
        for conv_id, added in by_conv.items():
            # Extend every cached window for this conversation instead of dropping it.
            for key in self._keys_for("msgs", conv_id):
                cached = self._cache.get(key)
                if cached is MISSING:
                    continue
                last = key[2]
                merged = list(cached) + added
                if last:
                    merged = merged[-last:]
                if len(merged) > self._max_messages:
                    self._cache.invalidate(key)  # too big to hold; the next read hits the database
                else:
                    self._cache.set(key, merged)
        self._invalidate_lists()

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        deleted = await self._inner.delete_conversation(conversation_id)
        self._forget(conversation_id)
        self._invalidate_lists()
        return deleted

    async def rename_conversation(self, conversation_id: UUID, title: str) -> bool:
        renamed = await self._inner.rename_conversation(conversation_id, title)
        self._cache.invalidate(("conv", conversation_id))
        self._invalidate_lists()
        return renamed

    def _keys_for(self, kind: str, conversation_id: UUID) -> list[Key]:
        return [k for k in self._cache.keys() if k[0] == kind and k[1] == conversation_id]

    def _forget(self, conversation_id: UUID) -> None:
        self._cache.invalidate_where(lambda k: len(k) > 1 and k[1] == conversation_id)

    def _invalidate_lists(self) -> None:
        self._cache.invalidate_where(lambda k: k[0] == "list")
