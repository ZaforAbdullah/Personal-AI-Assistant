"""Build the configured chat model and embedder from settings.

LM Studio uses `ChatOpenAI`. Gemini uses the native `ChatGoogleGenerativeAI`: Gemini 3 puts a
thought signature on every function call, the OpenAI-compatible endpoint's vendor field for it
is dropped by generic clients, and tool use then fails on the second turn. Embeddings use the
OpenAI-compatible endpoint for both providers.
"""

from __future__ import annotations

import inspect
import logging

from langchain_core.language_models import BaseChatModel

from app.config import Settings
from app.llm.embeddings import Embedder, OpenAICompatEmbedder

log = logging.getLogger(__name__)

# Where each provider library keeps the HTTP client(s) it opened for us.
#   ChatOpenAI:            root_async_client (AsyncOpenAI), root_client (OpenAI)
#   ChatGoogleGenerativeAI: async_client (aclose), client (close)
_CLIENT_ATTRS = ("root_async_client", "async_client", "root_client", "client")


async def close_chat_model(model: BaseChatModel) -> None:
    """Close the provider's HTTP clients at shutdown. Best effort: failures are logged."""
    for attr in _CLIENT_ATTRS:
        client = getattr(model, attr, None)
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is None:
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:
            log.warning("could not close %s of %s", attr, type(model).__name__, exc_info=True)


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            timeout=60,
            max_retries=1,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.lmstudio_model,
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key or "missing",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=60,
        max_retries=1,
        streaming=True,
        stream_usage=False,
    )


def build_embedder(settings: Settings) -> Embedder:
    model = settings.embedding_model
    # nomic-embed-text is asymmetric: it wants task prefixes on queries and documents.
    nomic = "nomic" in model.lower()
    return OpenAICompatEmbedder(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        dimensions=settings.embedding_dim,
        request_dimensions=settings.llm_provider == "gemini",
        document_prefix="search_document: " if nomic else "",
        query_prefix="search_query: " if nomic else "",
    )
