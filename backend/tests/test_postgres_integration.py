"""Integration tests against a real Postgres + pgvector.

They run whenever a Postgres is reachable: `TEST_DATABASE_URL` if set, otherwise the
compose default on localhost:5432 (`docker compose up -d db`). Without one they skip with
a reason, so the default suite is still offline-safe. They never touch the app's data: a
separate `assistant_test` database is created on first use and its tables are truncated after
each test.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.db.postgres import PostgresConversationRepository, PostgresKnowledgeRepository, init_db
from app.db.repositories import KnowledgeChunk, MessageRecord

COMPOSE_URL = "postgresql+asyncpg://assistant:assistant@localhost:5432/assistant"
TEST_DB = "assistant_test"


def _reachable(url: str) -> bool:
    parts = urlsplit(url)
    try:
        socket.create_connection((parts.hostname or "localhost", parts.port or 5432), 0.5).close()
        return True
    except OSError:
        return False


def _admin_url() -> str | None:
    """The URL to use, or None to skip. An explicit TEST_DATABASE_URL is never skipped:
    if it is unreachable the tests should fail loudly, not silently pass."""
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    return COMPOSE_URL if _reachable(COMPOSE_URL) else None


ADMIN_URL = _admin_url()

pytestmark = pytest.mark.skipif(
    ADMIN_URL is None,
    reason="no Postgres on localhost:5432 (run `docker compose up -d db`) and no TEST_DATABASE_URL",
)


def _with_database(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture(scope="session")
async def test_db_url() -> str:
    """Create the throwaway database once per session (idempotent)."""
    assert ADMIN_URL is not None
    admin = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB}
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    finally:
        await admin.dispose()
    return _with_database(ADMIN_URL, TEST_DB)


@pytest.fixture
async def engine(test_db_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(test_db_url)
    await init_db(engine)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE messages, conversations, knowledge_chunks"))
        await engine.dispose()


async def test_conversation_roundtrip(engine: AsyncEngine) -> None:
    repo = PostgresConversationRepository(engine)
    assert await repo.ping()
    conv = await repo.create_conversation()
    assert await repo.get_conversation(conv.id) is not None
    assert await repo.get_conversation(uuid4()) is None

    await repo.add_messages(
        [
            MessageRecord(conv.id, "user", "hi"),
            MessageRecord(
                conv.id,
                "assistant",
                "",
                tool_calls=[{"id": "c1", "name": "calculator", "arguments": "{}"}],
            ),
            MessageRecord(
                conv.id, "tool", '{"result": 4}', tool_call_id="c1", tool_name="calculator"
            ),
            MessageRecord(conv.id, "assistant", "4", status="complete"),
        ]
    )
    all_msgs = await repo.list_messages(conv.id)
    assert [m.role for m in all_msgs] == ["user", "assistant", "tool", "assistant"]
    assert all_msgs[1].tool_calls[0]["name"] == "calculator"
    last_two = await repo.list_messages(conv.id, last=2)
    assert [m.content for m in last_two] == ['{"result": 4}', "4"]

    summaries = await repo.list_conversations(limit=5)
    assert [s.id for s in summaries] == [conv.id]
    assert summaries[0].title == "hi" and summaries[0].message_count == 4

    assert await repo.rename_conversation(conv.id, "Renamed") is True
    assert (await repo.get_conversation(conv.id)).title == "Renamed"
    assert (await repo.list_conversations(limit=5))[0].title == "Renamed"
    assert await repo.rename_conversation(uuid4(), "nope") is False

    assert await repo.delete_conversation(conv.id) is True
    assert await repo.delete_conversation(conv.id) is False
    assert await repo.get_conversation(conv.id) is None
    assert await repo.list_messages(conv.id) == []  # cascaded
    assert await repo.list_conversations(limit=5) == []


async def test_knowledge_search_orders_by_cosine(engine: AsyncEngine) -> None:
    repo = PostgresKnowledgeRepository(engine)
    await repo.replace_all(
        [
            KnowledgeChunk(uuid4(), "a", "east", [1.0, 0.0, 0.0], "t"),
            KnowledgeChunk(uuid4(), "b", "north", [0.0, 1.0, 0.0], "t"),
            KnowledgeChunk(uuid4(), "c", "north-east", [0.7, 0.7, 0.0], "t"),
        ]
    )
    assert await repo.count("t") == 3
    hits = await repo.search([0.9, 0.1, 0.0], limit=2)
    assert [h.source for h in hits] == ["a", "c"]
    assert hits[0].score > hits[1].score


async def test_concurrent_sync_serialises_and_never_duplicates(engine: AsyncEngine) -> None:
    wanted = [("a", "east"), ("b", "north")]
    embeds = 0

    async def embed(texts: list[str]) -> list[list[float]]:
        nonlocal embeds
        embeds += 1
        await asyncio.sleep(0.1)  # hold the lock long enough for the others to queue
        return [[1.0, 0.0, 0.0] for _ in texts]

    # Separate repositories = separate sessions/connections, like separate replicas.
    repos = [PostgresKnowledgeRepository(engine) for _ in range(6)]
    results = await asyncio.gather(*(r.sync("t", wanted, embed) for r in repos))
    assert sorted(results) == [0] * 5 + [2]
    assert embeds == 1 and await repos[0].count("t") == 2

    # Edited text with the same row count is noticed; a failing embed rolls back cleanly.
    async def boom(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        await repos[0].sync("t", [("a", "east"), ("b", "changed")], boom)
    assert await repos[0].count("t") == 2
    assert await repos[1].sync("t", [("a", "east"), ("b", "changed")], embed) == 2
    assert await repos[2].sync("t", [("a", "east"), ("b", "changed")], embed) == 0


async def test_title_column_is_added_to_an_existing_table(engine: AsyncEngine) -> None:
    # init_db patches columns onto tables created by an older version (create_all alone
    # never alters existing tables). Simulate that by dropping the column and re-running.
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE conversations DROP COLUMN IF EXISTS title"))
    await init_db(engine)
    repo = PostgresConversationRepository(engine)
    conv = await repo.create_conversation()
    assert await repo.rename_conversation(conv.id, "works") is True
