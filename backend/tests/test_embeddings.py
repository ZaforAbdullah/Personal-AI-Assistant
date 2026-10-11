"""OpenAI-compatible embedder: dimension negotiation without a live provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from app.llm.embeddings import OpenAICompatEmbedder
from app.llm.errors import LLMError


class FakeEmbeddingsAPI:
    """Stands in for `AsyncOpenAI().embeddings`; records calls and scripts responses."""

    def __init__(
        self, width: int, *, reject_dimensions: bool = False, null_index: bool = False
    ) -> None:
        self.width = width
        self.reject_dimensions = reject_dimensions
        self.null_index = null_index  # Gemini's compatible endpoint sends index: null
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.reject_dimensions and "dimensions" in kwargs:
            request = httpx.Request("POST", "http://x/embeddings")
            response = httpx.Response(400, request=request, json={"error": "unknown dimensions"})
            raise openai.BadRequestError("unknown parameter", response=response, body=None)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=None if self.null_index else i, embedding=[float(i)] * self.width
                )
                for i, _ in enumerate(kwargs["input"])
            ]
        )


def make(api: FakeEmbeddingsAPI, *, request_dimensions: bool) -> OpenAICompatEmbedder:
    client = SimpleNamespace(embeddings=api)
    return OpenAICompatEmbedder(
        base_url="http://x",
        api_key="k",
        model="m",
        dimensions=768,
        request_dimensions=request_dimensions,
        query_prefix="q: ",
        client=client,  # type: ignore[arg-type]
    )


async def test_requests_dimensions_and_prefixes_queries() -> None:
    api = FakeEmbeddingsAPI(width=768)
    emb = make(api, request_dimensions=True)
    out = await emb.embed(["hello"], kind="query")
    assert len(out) == 1 and len(out[0]) == 768
    assert api.calls == [{"model": "m", "input": ["q: hello"], "dimensions": 768}]


async def test_adopts_native_width_when_provider_ignores_dimensions() -> None:
    api = FakeEmbeddingsAPI(width=3072)
    emb = make(api, request_dimensions=True)
    out = await emb.embed(["a", "b"])
    assert [len(v) for v in out] == [3072, 3072]
    assert emb.dimensions == 3072  # remembered, so the seed and queries agree


async def test_retries_without_dimensions_when_rejected() -> None:
    api = FakeEmbeddingsAPI(width=1024, reject_dimensions=True)
    emb = make(api, request_dimensions=True)
    out = await emb.embed(["a"])
    assert len(out[0]) == 1024
    assert "dimensions" in api.calls[0] and "dimensions" not in api.calls[1]
    await emb.embed(["b"])  # subsequent calls do not retry the rejected parameter
    assert "dimensions" not in api.calls[2]


async def test_tolerates_null_index_and_keeps_order() -> None:
    # Gemini's OpenAI-compatible /embeddings sends index=null on every item.
    api = FakeEmbeddingsAPI(width=8, null_index=True)
    emb = make(api, request_dimensions=True)
    out = await emb.embed(["a", "b", "c"])
    assert [v[0] for v in out] == [0.0, 1.0, 2.0]  # response order preserved


async def test_wrong_vector_count_is_an_llm_error() -> None:
    class Short:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0] * 8)])

    emb = make(Short(), request_dimensions=False)  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="1 vectors for 2 inputs"):
        await emb.embed(["a", "b"])


async def test_other_errors_surface_as_llm_error() -> None:
    class Down:
        async def create(self, **kwargs: Any) -> Any:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    emb = make(Down(), request_dimensions=False)  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="APIConnectionError"):
        await emb.embed(["a"])
