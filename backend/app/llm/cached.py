"""Embedding cache keyed by (model, kind, text); the win is repeated knowledge_search queries."""

from __future__ import annotations

from typing import Any

from app.cache import TTLCache
from app.llm.embeddings import Embedder, EmbedKind


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: TTLCache[Any, Any]) -> None:
        self._inner = inner
        self._cache = cache
        self.model = inner.model

    async def aclose(self) -> None:
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        keys = [("emb", self.model, kind, t) for t in texts]
        out: list[list[float] | None] = [self._cache.get(k, None) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            fresh = await self._inner.embed([texts[i] for i in missing], kind=kind)
            for i, vector in zip(missing, fresh, strict=True):
                out[i] = vector
                self._cache.set(keys[i], vector)
        return [v for v in out if v is not None]
