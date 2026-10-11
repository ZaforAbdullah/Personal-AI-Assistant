"""SQLAlchemy ORM tables. Same schema for local Postgres and Supabase."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.db.repositories import TITLE_MAX_CHARS


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ConversationRow(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # A user-chosen name; when NULL the first user message serves as the title.
    title: Mapped[str | None] = mapped_column(String(TITLE_MAX_CHARS), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MessageRow(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_seq", "conversation_id", "seq"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Insertion order is the source of truth for history; timestamps can tie within a batch.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(), unique=True)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="complete")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class KnowledgeChunkRow(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # untyped vector so EMBEDDING_DIM can change without a migration; rows are re-embedded together
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
