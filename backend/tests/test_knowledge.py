"""Knowledge base: chunking, seeding, and the knowledge_search tool. Offline via FakeEmbedder."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.db.repositories import InMemoryKnowledgeRepository
from app.db.seed import KNOWLEDGE_DIR, chunk_markdown, load_knowledge, seed_knowledge
from app.llm.embeddings import FakeEmbedder
from app.llm.errors import LLMError
from app.tools.base import ToolError
from app.tools.knowledge_search import KnowledgeSearchTool


def test_chunk_markdown_splits_on_headings_and_keeps_context() -> None:
    md = "# Title\n\nintro text\n\n## First\n\npara one\n\npara two\n\n## Second\n\nonly one\n"
    chunks = chunk_markdown("doc", md)
    assert [c.source for c in chunks] == ["doc", "doc#First", "doc#Second"]
    assert chunks[0].content == "Title\nintro text"
    assert chunks[1].content == "Title — First\npara one\n\npara two"


def test_chunk_markdown_splits_long_sections() -> None:
    md = "# T\n\n## S\n\n" + "\n\n".join("x" * 500 for _ in range(4))
    chunks = chunk_markdown("doc", md)
    assert len(chunks) == 2
    assert all(c.source == "doc#S" for c in chunks)


def test_bundled_knowledge_loads() -> None:
    chunks = load_knowledge(KNOWLEDGE_DIR)
    assert len(chunks) >= 10
    assert any("Portuguese" in c.content for c in chunks)


async def test_seed_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n## One\n\nhello\n\n## Two\n\nworld\n")
    repo, emb = InMemoryKnowledgeRepository(), FakeEmbedder()
    assert await seed_knowledge(repo, emb, tmp_path) == 2
    assert await seed_knowledge(repo, emb, tmp_path) == 0
    assert await repo.count(emb.model) == 2


async def test_seed_reembeds_when_model_changes(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n## One\n\nhello\n")
    repo = InMemoryKnowledgeRepository()
    await seed_knowledge(repo, FakeEmbedder(), tmp_path)

    class OtherEmbedder(FakeEmbedder):
        model = "other-model"

    assert await seed_knowledge(repo, OtherEmbedder(), tmp_path) == 1
    assert await repo.count("fake-embedder") == 0
    assert await repo.count("other-model") == 1


async def test_seed_notices_edited_text_even_when_the_count_is_unchanged(tmp_path: Path) -> None:
    doc = tmp_path / "a.md"
    doc.write_text("# A\n\n## One\n\nhello\n")
    repo, emb = InMemoryKnowledgeRepository(), FakeEmbedder()
    assert await seed_knowledge(repo, emb, tmp_path) == 1
    doc.write_text("# A\n\n## One\n\nhello, edited\n")
    assert await seed_knowledge(repo, emb, tmp_path) == 1
    assert [c.content for c in repo._chunks] == ["A — One\nhello, edited"]
    assert await seed_knowledge(repo, emb, tmp_path) == 0


async def test_concurrent_seeds_embed_once_and_store_no_duplicates(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n## One\n\nhello\n\n## Two\n\nworld\n")
    calls = 0

    class SlowEmbedder(FakeEmbedder):
        async def embed(self, texts, *, kind="document"):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)  # widen the race window
            return await super().embed(texts, kind=kind)

    repo, emb = InMemoryKnowledgeRepository(), SlowEmbedder()
    results = await asyncio.gather(*(seed_knowledge(repo, emb, tmp_path) for _ in range(8)))
    assert sorted(results) == [0] * 7 + [2]
    assert calls == 1 and len(repo._chunks) == 2


async def test_failed_seed_keeps_the_previous_contents(tmp_path: Path) -> None:
    doc = tmp_path / "a.md"
    doc.write_text("# A\n\n## One\n\nhello\n")
    repo = InMemoryKnowledgeRepository()
    await seed_knowledge(repo, FakeEmbedder(), tmp_path)
    doc.write_text("# A\n\n## One\n\nchanged\n")

    class DownEmbedder(FakeEmbedder):
        async def embed(self, texts, *, kind="document"):
            raise LLMError("connection refused")

    with pytest.raises(LLMError):
        await seed_knowledge(repo, DownEmbedder(), tmp_path)
    assert [c.content for c in repo._chunks] == ["A — One\nhello"]  # old data still served


async def test_search_returns_relevant_chunk_first(tmp_path: Path) -> None:
    (tmp_path / "geo.md").write_text(
        "# Geo\n\n## Markets\n\nUsers live in Brazil, Mexico and Colombia.\n\n"
        "## Office\n\nThe engineering office is in Madrid.\n"
    )
    (tmp_path / "tools.md").write_text("# Tools\n\n## Calculator\n\nEvaluates arithmetic.\n")
    repo, emb = InMemoryKnowledgeRepository(), FakeEmbedder()
    await seed_knowledge(repo, emb, tmp_path)
    tool = KnowledgeSearchTool(embedder=emb, repo=repo, limit=2)

    out = await tool.ainvoke(dict(query="Which markets have users in Brazil?"))
    assert isinstance(out, dict)
    assert [r["source"] for r in out["results"]] == ["geo#Markets", "geo#Office"]
    assert out["results"][0]["score"] > out["results"][1]["score"]
    assert "Brazil" in out["results"][0]["content"]


async def test_search_over_bundled_docs_finds_markets() -> None:
    repo, emb = InMemoryKnowledgeRepository(), FakeEmbedder()
    await seed_knowledge(repo, emb, KNOWLEDGE_DIR)
    tool = KnowledgeSearchTool(embedder=emb, repo=repo, limit=3)
    out = await tool.ainvoke(
        dict(query="Which languages does the assistant reply in? Portuguese Spanish")
    )
    assert isinstance(out, dict)
    assert "assistant-overview#Languages" in [r["source"] for r in out["results"]]


async def test_search_on_empty_store_is_a_tool_error() -> None:
    tool = KnowledgeSearchTool(embedder=FakeEmbedder(), repo=InMemoryKnowledgeRepository())
    with pytest.raises(ToolError, match="empty"):
        await tool.ainvoke(dict(query="anything at all"))


async def test_search_reports_embedding_outage_as_tool_error() -> None:
    class DownEmbedder(FakeEmbedder):
        async def embed(self, texts, *, kind="document"):
            raise LLMError("connection refused")

    tool = KnowledgeSearchTool(embedder=DownEmbedder(), repo=InMemoryKnowledgeRepository())
    with pytest.raises(ToolError, match="embedding service unavailable"):
        await tool.ainvoke(dict(query="anything at all"))


def test_schema_exposes_query() -> None:
    schema = convert_to_openai_tool(
        KnowledgeSearchTool(embedder=FakeEmbedder(), repo=InMemoryKnowledgeRepository())
    )
    assert schema["function"]["parameters"]["required"] == ["query"]
