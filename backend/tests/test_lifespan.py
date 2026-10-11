"""Lifespan: startup acquires, shutdown releases, and shutdown drains in-flight work.

Follows FastAPI's lifespan guidance (https://fastapi.tiangolo.com/advanced/events/): code
before `yield` is startup, code after it is shutdown, and shutdown must run for whatever
startup managed to acquire, even when startup fails half-way.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.db.repositories import InMemoryConversationRepository, InMemoryKnowledgeRepository
from app.llm.embeddings import FakeEmbedder
from app.llm.fake import ScriptedChatModel
from app.main import create_app
from tests.conftest import WatchedModel


class Spy(httpx.AsyncClient):
    """An httpx client that records being closed."""

    instances: list[Spy] = []

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.closed_by_test = False
        Spy.instances.append(self)

    async def aclose(self) -> None:
        self.closed_by_test = True
        await super().aclose()


@pytest.fixture
def spy_client(monkeypatch: pytest.MonkeyPatch) -> type[Spy]:
    Spy.instances.clear()
    monkeypatch.setattr("app.main.make_client", lambda: Spy())
    return Spy


def in_memory_app(settings: Settings, **kw: Any):
    return create_app(
        settings,
        embedder=kw.pop("embedder", FakeEmbedder()),
        conversations=InMemoryConversationRepository(),
        knowledge=InMemoryKnowledgeRepository(),
        **kw,
    )


async def test_shutdown_closes_the_http_client_and_drops_state(spy_client: type[Spy]) -> None:
    app = in_memory_app(Settings(_env_file=None), chat_model=ScriptedChatModel())
    async with app.router.lifespan_context(app):
        assert hasattr(app.state, "container")
        assert not spy_client.instances[0].closed_by_test
    assert spy_client.instances[0].closed_by_test
    # references to the graph, cache and clients are dropped
    assert not hasattr(app.state, "container")


async def test_failed_startup_still_releases_what_was_acquired(
    spy_client: type[Spy], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The HTTP client (and everything before it) already exists when the last startup step
    # blows up. A bare try/finally around the `yield` never runs in that case; the exit
    # stack must still release what was acquired.
    def exploding_service(**kwargs: Any) -> None:
        raise RuntimeError("startup failed half-way")

    monkeypatch.setattr("app.main.ChatService", exploding_service)
    app = in_memory_app(Settings(_env_file=None), chat_model=ScriptedChatModel())
    with pytest.raises(RuntimeError, match="half-way"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup should not have completed")
    assert spy_client.instances and all(c.closed_by_test for c in spy_client.instances)
    assert not hasattr(app.state, "container")


async def test_bad_configuration_fails_before_acquiring_anything(spy_client: type[Spy]) -> None:
    settings = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="")
    app = in_memory_app(settings)  # no injected chat model -> the factory raises
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup should not have completed")
    assert all(c.closed_by_test for c in spy_client.instances)  # whatever existed was closed


class SpyEngine:
    disposed = False

    async def dispose(self) -> None:
        SpyEngine.disposed = True


@pytest.fixture
def fake_postgres(monkeypatch: pytest.MonkeyPatch) -> type[SpyEngine]:
    SpyEngine.disposed = False

    async def no_init(engine: object) -> None:
        return None

    monkeypatch.setattr("app.main.create_engine", lambda s: SpyEngine())
    monkeypatch.setattr("app.main.init_db", no_init)
    monkeypatch.setattr(
        "app.main.PostgresConversationRepository", lambda e: InMemoryConversationRepository()
    )
    monkeypatch.setattr(
        "app.main.PostgresKnowledgeRepository", lambda e: InMemoryKnowledgeRepository()
    )
    return SpyEngine


async def test_engine_is_disposed_on_shutdown(fake_postgres: type[SpyEngine]) -> None:
    app = create_app(
        Settings(_env_file=None), chat_model=ScriptedChatModel(), embedder=FakeEmbedder()
    )
    async with app.router.lifespan_context(app):
        assert fake_postgres.disposed is False
    assert fake_postgres.disposed is True


async def test_engine_is_disposed_even_when_startup_fails(
    fake_postgres: type[SpyEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding_service(**kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("app.main.ChatService", exploding_service)
    app = create_app(
        Settings(_env_file=None), chat_model=ScriptedChatModel(), embedder=FakeEmbedder()
    )
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    assert fake_postgres.disposed is True


async def test_only_owned_resources_are_closed(spy_client: type[Spy]) -> None:
    class Injected(FakeEmbedder):
        closed = False

        async def aclose(self) -> None:
            Injected.closed = True

    injected = Injected()
    app = in_memory_app(Settings(_env_file=None), chat_model=ScriptedChatModel(), embedder=injected)
    async with app.router.lifespan_context(app):
        pass
    assert Injected.closed is False  # the caller's collaborator is the caller's to close


async def test_owned_provider_clients_are_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class OwnedEmbedder(FakeEmbedder):
        async def aclose(self) -> None:
            closed.append("embedder")

    class OwnedModel(ScriptedChatModel):
        pass

    async def close_model(model: object) -> None:
        closed.append("model")

    monkeypatch.setattr("app.main.build_embedder", lambda s: OwnedEmbedder())
    monkeypatch.setattr("app.main.build_chat_model", lambda s: OwnedModel())
    monkeypatch.setattr("app.main.close_chat_model", close_model)
    app = create_app(
        Settings(_env_file=None),
        conversations=InMemoryConversationRepository(),
        knowledge=InMemoryKnowledgeRepository(),
    )
    async with app.router.lifespan_context(app):
        assert closed == []
    assert sorted(closed) == ["embedder", "model"]


class SlowWrites(InMemoryConversationRepository):
    """Writes take a moment, like a real database round trip."""

    async def add_messages(self, messages):  # type: ignore[override]
        await asyncio.sleep(0.05)
        await super().add_messages(messages)


async def test_shutdown_waits_for_in_flight_turns_and_stores_their_partial_replies() -> None:
    """What uvicorn does on SIGTERM once --timeout-graceful-shutdown expires: cancel every
    request task, then run the lifespan shutdown *without waiting for those tasks*. The
    partial reply of a turn that was mid-stream must still reach the database, which means
    shutdown has to drain in-flight work before it disposes the resources it needs."""
    repo = SlowWrites()
    WatchedModel.closed = False
    app = create_app(
        Settings(_env_file=None),
        chat_model=WatchedModel(),
        embedder=FakeEmbedder(),
        conversations=repo,
        knowledge=InMemoryKnowledgeRepository(),
    )
    body = json.dumps({"message": "tell me a long story"}).encode()
    sent: list[dict[str, Any]] = []
    streaming = asyncio.Event()
    never = asyncio.Event()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await never.wait()  # a client that stays connected, like uvicorn's receive()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if sum(m["type"] == "http.response.body" for m in sent) >= 6:
            streaming.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/chat",
        "raw_path": b"/api/chat",
        "query_string": b"",
        "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }

    async with app.router.lifespan_context(app):
        request = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(streaming.wait(), timeout=5)
        request.cancel()  # what uvicorn does when graceful shutdown times out
        # the lifespan shutdown then starts at once, without awaiting `request`
    await asyncio.gather(request, return_exceptions=True)

    convs = await repo.list_conversations()
    assert len(convs) == 1
    stored = await repo.list_messages(convs[0].id)
    assert [m.role for m in stored] == ["user", "assistant"], "the partial reply was lost"
    assert stored[1].status == "interrupted" and stored[1].content.startswith("tok")
    # LangGraph cancels its node task a loop turn or two after the stream is closed, so allow
    # a short settle; what matters is that it is closed promptly, not at process exit.
    for _ in range(100):
        if WatchedModel.closed:
            break
        await asyncio.sleep(0.01)
    assert WatchedModel.closed, "the upstream model stream was left open"
