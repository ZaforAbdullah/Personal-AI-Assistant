"""knowledge_search tool: semantic lookup over the seeded knowledge base (pgvector)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.llm.errors import LLMError
from app.tools.base import AsyncTool, ToolError


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(
        description="What to look up, phrased as a short natural-language question.",
        min_length=2,
        max_length=500,
    )


class KnowledgeSearchTool(AsyncTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "knowledge_search"
    description: str = (
        "Search the knowledge base about this assistant (what it does, languages, models) and "
        "about how this chat app is built (its tools, providers, design). Returns the most "
        "relevant passages with a similarity score."
    )
    args_schema: type[BaseModel] = KnowledgeSearchArgs

    # Protocols can't be validated by pydantic, so these are typed loosely here.
    embedder: Any = Field(exclude=True)  # Embedder
    repo: Any = Field(exclude=True)  # KnowledgeRepository
    limit: int = 4

    async def _arun(self, query: str) -> dict[str, Any]:
        try:
            [vector] = await self.embedder.embed([query], kind="query")
        except LLMError as exc:
            raise ToolError(f"embedding service unavailable: {exc}") from exc
        hits = await self.repo.search(vector, limit=self.limit)
        if not hits:
            raise ToolError("the knowledge base is empty")
        return {
            "query": query,
            "results": [
                {"source": h.source, "score": round(h.score, 3), "content": h.content} for h in hits
            ],
        }
