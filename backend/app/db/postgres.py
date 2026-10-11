"""Postgres (+ pgvector) implementations of the repository Protocols."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import delete, exists, func, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Base, ConversationRow, KnowledgeChunkRow, MessageRow
from app.db.repositories import (
    TITLE_MAX_CHARS,
    ConversationRecord,
    ConversationSummary,
    EmbedTexts,
    KnowledgeChunk,
    KnowledgeHit,
    MessageRecord,
    WantedChunks,
    _build_chunks,
    make_title,
)

log = logging.getLogger(__name__)

# Arbitrary constant naming the "seed the knowledge base" critical section (bigint).
_SEED_LOCK_KEY = 0x4C555A4B42  # "LUZKB"


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.db_url,
        connect_args=settings.db_connect_args,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


async def init_db(engine: AsyncEngine) -> None:
    """Enable pgvector and create tables. (A migration tool would replace this in prod.)"""
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except DBAPIError as exc:
            # Managed Postgres (e.g. Supabase) may not let the app role do this; the
            # extension can be enabled from the dashboard instead.
            log.warning("could not create pgvector extension (%s); assuming it exists", exc)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all never alters existing tables. Columns added after the first release
        # are patched in here; this is the point where a real migration tool takes over.
        await conn.execute(
            text(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS "
                f"title VARCHAR({TITLE_MAX_CHARS})"
            )
        )


class PostgresConversationRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def create_conversation(self) -> ConversationRecord:
        async with self._sessions.begin() as session:
            row = ConversationRow()
            session.add(row)
            await session.flush()
            return ConversationRecord(id=row.id, created_at=row.created_at, title=row.title)

    async def get_conversation(self, conversation_id: UUID) -> ConversationRecord | None:
        async with self._sessions() as session:
            row = await session.get(ConversationRow, conversation_id)
            if row is None:
                return None
            return ConversationRecord(id=row.id, created_at=row.created_at, title=row.title)

    async def add_messages(self, messages: Iterable[MessageRecord]) -> None:
        rows = [
            MessageRow(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                tool_name=m.tool_name,
                status=m.status,
                created_at=m.created_at,
            )
            for m in messages
        ]
        if not rows:
            return
        async with self._sessions.begin() as session:
            session.add_all(rows)

    async def list_messages(
        self, conversation_id: UUID, *, last: int | None = None
    ) -> list[MessageRecord]:
        async with self._sessions() as session:
            stmt = select(MessageRow).where(MessageRow.conversation_id == conversation_id)
            if last:
                stmt = stmt.order_by(MessageRow.seq.desc()).limit(last)
                rows = list(reversed((await session.scalars(stmt)).all()))
            else:
                rows = list((await session.scalars(stmt.order_by(MessageRow.seq))).all())
        return [
            MessageRecord(
                id=r.id,
                conversation_id=r.conversation_id,
                role=r.role,  # type: ignore[arg-type]
                content=r.content,
                tool_calls=r.tool_calls,
                tool_call_id=r.tool_call_id,
                tool_name=r.tool_name,
                status=r.status,  # type: ignore[arg-type]
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        m = MessageRow
        first_user = (
            select(m.content)
            .where(m.conversation_id == ConversationRow.id, m.role == "user")
            .order_by(m.seq)
            .limit(1)
            .scalar_subquery()
        )
        last_at = (
            select(func.max(m.created_at))
            .where(m.conversation_id == ConversationRow.id)
            .scalar_subquery()
        )
        count = (
            select(func.count()).where(m.conversation_id == ConversationRow.id).scalar_subquery()
        )
        stmt = (
            select(
                ConversationRow.id,
                ConversationRow.created_at,
                ConversationRow.title.label("custom_title"),
                first_user.label("first_user"),
                last_at.label("updated_at"),
                count.label("message_count"),
            )
            .where(exists().where(m.conversation_id == ConversationRow.id))
            .order_by(last_at.desc())
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(stmt)).all()
        return [
            ConversationSummary(
                id=row.id,
                title=row.custom_title or make_title(row.first_user or "") or "New chat",
                created_at=row.created_at,
                updated_at=row.updated_at,
                message_count=int(row.message_count),
            )
            for row in rows
        ]

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        async with self._sessions.begin() as session:
            # messages go with it via ON DELETE CASCADE
            result = await session.execute(
                delete(ConversationRow).where(ConversationRow.id == conversation_id)
            )
            return bool(result.rowcount)

    async def rename_conversation(self, conversation_id: UUID, title: str) -> bool:
        async with self._sessions.begin() as session:
            result = await session.execute(
                update(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .values(title=title)
            )
            return bool(result.rowcount)

    async def ping(self) -> bool:
        async with self._sessions() as session:
            return (await session.execute(text("SELECT 1"))).scalar() == 1


class PostgresKnowledgeRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def count(self, embedding_model: str) -> int:
        async with self._sessions() as session:
            stmt = select(func.count()).where(KnowledgeChunkRow.embedding_model == embedding_model)
            return int((await session.execute(stmt)).scalar_one())

    async def replace_all(self, chunks: Iterable[KnowledgeChunk]) -> None:
        async with self._sessions.begin() as session:
            await self._replace(session, chunks)

    @staticmethod
    async def _replace(session, chunks: Iterable[KnowledgeChunk]) -> None:
        await session.execute(delete(KnowledgeChunkRow))
        session.add_all(
            KnowledgeChunkRow(
                id=c.id,
                source=c.source,
                content=c.content,
                embedding=c.embedding,
                embedding_model=c.embedding_model,
            )
            for c in chunks
        )

    async def sync(self, embedding_model: str, wanted: WantedChunks, embed: EmbedTexts) -> int:
        # One transaction under an advisory lock. Without it, two replicas starting together both
        # see stale data, both embed and both insert (the second DELETE can't see the first's rows).
        # With it, the second waits, sees the result and does nothing. The lock is released on
        # commit, rollback or a dropped connection. If embed() raises, the old rows survive.
        async with self._sessions.begin() as session:
            await session.execute(select(func.pg_advisory_xact_lock(_SEED_LOCK_KEY)))
            rows = await session.execute(
                select(
                    KnowledgeChunkRow.embedding_model,
                    KnowledgeChunkRow.source,
                    KnowledgeChunkRow.content,
                )
            )
            have = Counter(tuple(r) for r in rows.all())
            if have == Counter((embedding_model, s, c) for s, c in wanted):
                return 0
            vectors = await embed([c for _, c in wanted])
            chunks = _build_chunks(embedding_model, wanted, vectors)
            await self._replace(session, chunks)
            return len(chunks)

    async def search(self, embedding: list[float], *, limit: int) -> list[KnowledgeHit]:
        distance = KnowledgeChunkRow.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(KnowledgeChunkRow.source, KnowledgeChunkRow.content, distance)
            .order_by(distance)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(stmt)).all()
        return [
            KnowledgeHit(source=source, content=content, score=1.0 - float(dist))
            for source, content, dist in rows
        ]
