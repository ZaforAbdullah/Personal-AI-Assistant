"""The in-process cache and the two things it fronts: conversation reads and embeddings."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.cache import MISSING, TTLCache
from app.db.cached import CachedConversationRepository
from app.db.repositories import InMemoryConversationRepository, MessageRecord
from app.llm.cached import CachedEmbedder
from app.llm.embeddings import FakeEmbedder


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_ttl_cache_hits_misses_expiry_and_lru() -> None:
    clock = Clock()
    cache: TTLCache[str, int] = TTLCache(max_entries=2, ttl_s=10, clock=clock)
    assert bool(cache) is True and len(cache) == 0  # empty is not "absent"
    assert cache.get("a") is MISSING
    cache.set("a", 1)
    assert cache.get("a") == 1
    clock.now += 11
    assert cache.get("a") is MISSING  # expired
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # touch a so b is the least recently used
    cache.set("c", 3)  # over capacity: evicts b
    assert cache.get("b") is MISSING and cache.get("a") == 1 and cache.get("c") == 3
    assert cache.stats.hits == 4 and cache.stats.misses == 3 and cache.stats.entries == 2
    assert cache.invalidate_where(lambda k: k in ("a", "zzz")) == 1
    assert len(cache) == 1


class CountingRepo(InMemoryConversationRepository):
    """Counts database reads so tests can prove the cache absorbed them."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def list_messages(self, conversation_id, *, last=None):
        self.reads += 1
        return await super().list_messages(conversation_id, last=last)

    async def get_conversation(self, conversation_id):
        self.reads += 1
        return await super().get_conversation(conversation_id)

    async def list_conversations(self, *, limit=50):
        self.reads += 1
        return await super().list_conversations(limit=limit)


async def test_chat_turn_history_is_served_from_cache_after_first_load() -> None:
    inner = CountingRepo()
    repo = CachedConversationRepository(inner, TTLCache(max_entries=100, ttl_s=60))
    conv = await repo.create_conversation()

    # First turn: user message written, window read (the cache already knows this
    # conversation is empty, so even the first read needs no query).
    await repo.add_messages([MessageRecord(conv.id, "user", "hi")])
    assert [m.content for m in await repo.list_messages(conv.id, last=30)] == ["hi"]
    await repo.add_messages([MessageRecord(conv.id, "assistant", "hello")])
    assert inner.reads == 0

    # Second turn: still no database read; the window was extended in place.
    await repo.add_messages([MessageRecord(conv.id, "user", "again")])
    window = await repo.list_messages(conv.id, last=30)
    assert [m.content for m in window] == ["hi", "hello", "again"]
    assert inner.reads == 0
    # And it matches what the database holds (peek at storage without counting a read).
    assert [m.content for m in inner._messages[conv.id]] == ["hi", "hello", "again"]

    # A window bound is respected after in-place extension.
    assert [m.content for m in await repo.list_messages(conv.id, last=2)] == ["hello", "again"]
    await repo.add_messages([MessageRecord(conv.id, "assistant", "sure")])
    assert [m.content for m in await repo.list_messages(conv.id, last=2)] == ["again", "sure"]
    assert inner.reads == 0


async def test_cold_conversation_reads_once_then_hits() -> None:
    inner = CountingRepo()
    conv = await inner.create_conversation()  # created behind the cache's back
    await inner.add_messages([MessageRecord(conv.id, "user", "old")])
    repo = CachedConversationRepository(inner, TTLCache(max_entries=100, ttl_s=60))

    assert (await repo.get_conversation(conv.id)) is not None
    assert (await repo.get_conversation(conv.id)) is not None
    assert [m.content for m in await repo.list_messages(conv.id, last=30)] == ["old"]
    assert [m.content for m in await repo.list_messages(conv.id, last=30)] == ["old"]
    assert inner.reads == 2  # one per key, not one per call
    assert await repo.get_conversation(uuid4()) is None  # misses are not cached


async def test_writes_invalidate_lists_and_deletes_forget_everything() -> None:
    inner = CountingRepo()
    repo = CachedConversationRepository(inner, TTLCache(max_entries=100, ttl_s=60))
    a = await repo.create_conversation()
    await repo.add_messages([MessageRecord(a.id, "user", "first")])

    assert [s.title for s in await repo.list_conversations()] == ["first"]
    await repo.list_conversations()
    assert inner.reads == 1  # second list call was a hit

    assert await repo.rename_conversation(a.id, "Renamed")
    assert [s.title for s in await repo.list_conversations()] == ["Renamed"]  # list refetched
    assert inner.reads == 2

    assert await repo.delete_conversation(a.id)
    assert await repo.get_conversation(a.id) is None
    assert await repo.list_messages(a.id) == []
    assert await repo.list_conversations() == []


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    async def embed(self, texts, *, kind="document"):
        self.calls.append(list(texts))
        return await super().embed(texts, kind=kind)


async def test_embedder_only_embeds_misses_and_keeps_order() -> None:
    inner = CountingEmbedder()
    emb = CachedEmbedder(inner, TTLCache(max_entries=100, ttl_s=60))
    first = await emb.embed(["a", "b"], kind="query")
    second = await emb.embed(["b", "c", "a"], kind="query")
    assert inner.calls == [["a", "b"], ["c"]]
    assert second[0] == first[1] and second[2] == first[0]
    assert second[1] == (await inner.embed(["c"], kind="query"))[0]
    # Different kind is a different key (asymmetric models embed queries differently).
    await emb.embed(["a"], kind="document")
    assert inner.calls[-1] == ["a"]
    assert emb.model == inner.model


def test_expired_entries_are_swept_not_just_hidden() -> None:
    clock = Clock()
    cache: TTLCache[str, int] = TTLCache(max_entries=10_000, ttl_s=10, clock=clock)
    for i in range(500):
        cache.set(f"old{i}", i)
    clock.now += 11  # all of them expired; nobody ever reads them again
    for i in range(600):
        cache.set(f"new{i}", i)  # ordinary writes must reclaim the dead entries
    assert len(cache) <= 700
    assert cache.get("old0", None) is None


async def test_full_history_entries_do_not_grow_without_bound() -> None:
    inner = InMemoryConversationRepository()
    cache: TTLCache[Any, Any] = TTLCache(max_entries=100, ttl_s=60)
    repo = CachedConversationRepository(inner, cache, max_cached_messages=50)
    conv = await repo.create_conversation()
    for i in range(200):
        await repo.add_messages([MessageRecord(conv.id, "user", f"m{i}")])
    held = max((len(v) for v in cache._data.values() if isinstance(v[1], list)), default=0)
    assert held <= 50, f"cache holds {held} messages for one conversation"
    # Correctness is unaffected: an uncached read falls through to the database.
    assert len(await repo.list_messages(conv.id)) == 200
    assert [m.content for m in await repo.list_messages(conv.id, last=2)] == ["m198", "m199"]
