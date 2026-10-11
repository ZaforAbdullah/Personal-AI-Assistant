"""Embedding clients: the OpenAI-compatible `/embeddings` endpoint, plus a fake for tests."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Literal, Protocol

import openai
from openai import AsyncOpenAI

from app.llm.errors import LLMError

log = logging.getLogger(__name__)

EmbedKind = Literal["document", "query"]


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        """Embed texts. `kind` lets asymmetric models prefix queries and documents differently."""
        ...


class OpenAICompatEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        request_dimensions: bool,
        document_prefix: str = "",
        query_prefix: str = "",
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        # preferred width: Gemini honours `dimensions`, others return their native width;
        # whatever comes back is stored (rows are embedded together, so they never mix)
        self.dimensions = dimensions
        self._request_dimensions = request_dimensions
        self._prefix = {"document": document_prefix, "query": query_prefix}
        # Only close a client we created; an injected one belongs to whoever injected it.
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url, api_key=api_key or "missing", timeout=60.0, max_retries=1
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        if not texts:
            return []
        prefix = self._prefix[kind]
        inputs = [prefix + t for t in texts]
        try:
            resp = await self._create(inputs, with_dimensions=self._request_dimensions)
        except openai.BadRequestError as exc:
            if not self._request_dimensions:
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc
            # endpoint rejected `dimensions`: retry without it and keep the native width
            log.warning(
                "embedding endpoint rejected dimensions=%d; retrying without", self.dimensions
            )
            self._request_dimensions = False
            try:
                resp = await self._create(inputs, with_dimensions=False)
            except openai.OpenAIError as exc2:
                raise LLMError(f"{type(exc2).__name__}: {exc2}") from exc2
        except openai.OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        items = list(resp.data)
        # OpenAI numbers items with `index`; Gemini's compatible endpoint sends null and
        # relies on order. Sort only when every index is present.
        if all(getattr(item, "index", None) is not None for item in items):
            items.sort(key=lambda d: d.index)
        vectors = [list(item.embedding) for item in items]
        if len(vectors) != len(inputs):
            raise LLMError(
                f"embedding model {self.model} returned {len(vectors)} vectors for "
                f"{len(inputs)} inputs"
            )
        widths = {len(v) for v in vectors}
        if len(widths) != 1:
            raise LLMError(f"embedding model {self.model} returned mixed widths {sorted(widths)}")
        (width,) = widths
        if width != self.dimensions:
            log.warning(
                "embedding model %s returns %d dims (EMBEDDING_DIM=%d); using %d",
                self.model,
                width,
                self.dimensions,
                width,
            )
            self.dimensions = width
        return vectors

    async def _create(self, inputs: list[str], *, with_dimensions: bool):
        kwargs: dict[str, object] = {"model": self.model, "input": inputs}
        if with_dimensions:
            kwargs["dimensions"] = self.dimensions
        return await self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]


class FakeEmbedder:
    """Deterministic bag-of-words embedding (tokens of 4+ chars, so stopwords don't dominate).
    Similar texts get similar vectors, offline."""

    model = "fake-embedder"

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]{4,}", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324 (not security)
            vec[h % self.dimensions] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
