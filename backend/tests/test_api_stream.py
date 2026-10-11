"""HTTP layer: SSE framing, heartbeats, 404s, history endpoint. Runs against the ASGI app
in-process with a scripted LLM and in-memory storage."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel

from app.agent.events import TextDeltaEvent
from app.config import Settings
from app.db.repositories import InMemoryConversationRepository, InMemoryKnowledgeRepository
from app.llm.embeddings import FakeEmbedder
from app.llm.fake import ScriptedChatModel, text_turn, tool_turn
from app.main import create_app
from app.sse import sse_stream


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip() or frame.startswith(":"):
            continue
        name = data = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        events.append((name, data))
    return events


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, sse_heartbeat_s=15, tool_timeout_s=1)


async def make_client(settings: Settings, model: ScriptedChatModel, repo=None):
    """App wired entirely in-memory: scripted LLM, fake embedder, no database."""
    app = create_app(
        settings,
        chat_model=model,
        embedder=FakeEmbedder(),
        conversations=repo or InMemoryConversationRepository(),
        knowledge=InMemoryKnowledgeRepository(),
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return app, client


async def test_chat_streams_typed_events(settings: Settings) -> None:
    model = ScriptedChatModel(
        turns=[tool_turn("calculator", '{"expression": "2+2"}', call_id="c"), text_turn("It's 4.")]
    )
    app, client = await make_client(settings, model)
    async with app.router.lifespan_context(app), client:
        resp = await client.post("/api/chat", json={"message": "2+2?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(resp.text)

    names = [n for n, _ in events]
    assert names[0] == "conversation"
    assert names[1:3] == ["tool_start", "tool_result"]
    assert names[-1] == "done"
    assert "".join(d["text"] for n, d in events if n == "text_delta") == "It's 4."
    tool_start = next(d for n, d in events if n == "tool_start")
    assert tool_start == {
        "type": "tool_start",
        "call_id": "c",
        "name": "calculator",
        "args": {"expression": "2+2"},
    }


async def test_locale_sets_the_reply_language(settings: Settings) -> None:
    model = ScriptedChatModel(turns=[text_turn("Olá!"), text_turn("Hi!")])
    app, client = await make_client(settings, model)
    async with app.router.lifespan_context(app), client:
        await client.post("/api/chat", json={"message": "oi", "locale": "pt-BR"})
        await client.post("/api/chat", json={"message": "hi"})  # no locale -> English
        bad = await client.post("/api/chat", json={"message": "x", "locale": "not a tag!"})
    assert "Always reply in Portuguese" in model.calls[0]["messages"][0].text
    assert "Always reply in English" in model.calls[1]["messages"][0].text
    assert bad.status_code == 422


async def test_history_roundtrip(settings: Settings) -> None:
    model = ScriptedChatModel(turns=[text_turn("hi!"), text_turn("again")])
    app, client = await make_client(settings, model)
    async with app.router.lifespan_context(app), client:
        first = parse_sse((await client.post("/api/chat", json={"message": "hello"})).text)
        conv_id = first[0][1]["conversation_id"]
        await client.post("/api/chat", json={"message": "more", "conversation_id": conv_id})

        resp = await client.get(f"/api/conversations/{conv_id}/messages")
        assert resp.status_code == 200
        msgs = resp.json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in msgs] == ["hello", "hi!", "more", "again"]


async def test_conversation_list_and_delete(settings: Settings) -> None:
    model = ScriptedChatModel(turns=[text_turn("a"), text_turn("b"), text_turn("c")])
    app, client = await make_client(settings, model)
    async with app.router.lifespan_context(app), client:
        assert (await client.get("/api/conversations")).json() == []

        first = parse_sse((await client.post("/api/chat", json={"message": "first chat"})).text)
        first_id = first[0][1]["conversation_id"]
        second = parse_sse(
            (await client.post("/api/chat", json={"message": "  second   chat  "})).text
        )
        second_id = second[0][1]["conversation_id"]
        # a follow-up bumps the first conversation back to the top
        await client.post("/api/chat", json={"message": "again", "conversation_id": first_id})

        convs = (await client.get("/api/conversations")).json()
        assert [c["id"] for c in convs] == [first_id, second_id]
        assert [c["title"] for c in convs] == ["first chat", "second chat"]
        assert [c["message_count"] for c in convs] == [4, 2]
        assert convs[0]["updated_at"] >= convs[1]["updated_at"]

        assert (await client.get("/api/conversations?limit=1")).json()[0]["id"] == first_id
        assert (await client.get("/api/conversations?limit=0")).status_code == 422

        # rename: whitespace collapsed, echoed back, and reflected in the list
        renamed = await client.patch(
            f"/api/conversations/{second_id}", json={"title": "  My   second  chat "}
        )
        assert renamed.status_code == 200 and renamed.json()["title"] == "My second chat"
        titles = {c["id"]: c["title"] for c in (await client.get("/api/conversations")).json()}
        assert titles[second_id] == "My second chat" and titles[first_id] == "first chat"
        assert (
            await client.patch(f"/api/conversations/{second_id}", json={"title": "   "})
        ).status_code == 422
        assert (
            await client.patch(f"/api/conversations/{second_id}", json={"title": "x" * 81})
        ).status_code == 422
        assert (
            await client.patch(f"/api/conversations/{uuid4()}", json={"title": "ghost"})
        ).status_code == 404

        assert (await client.delete(f"/api/conversations/{second_id}")).status_code == 204
        assert (await client.delete(f"/api/conversations/{second_id}")).status_code == 404
        assert (await client.get(f"/api/conversations/{second_id}/messages")).status_code == 404
        assert [c["id"] for c in (await client.get("/api/conversations")).json()] == [first_id]


async def test_unknown_conversation_is_404(settings: Settings) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        resp = await client.post(
            "/api/chat", json={"message": "x", "conversation_id": str(uuid4())}
        )
        assert resp.status_code == 404
        resp = await client.get(f"/api/conversations/{uuid4()}/messages")
        assert resp.status_code == 404


async def preflight(client: httpx.AsyncClient, origin: str) -> httpx.Response:
    return await client.options(
        "/api/chat",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
        "http://0.0.0.0:3000",
        "http://192.168.1.42:3000",
        "https://my-laptop.local:3000",
    ],
)
async def test_cors_allows_any_host_on_the_frontend_port(settings: Settings, origin: str) -> None:
    # Whichever hostname a browser uses for "this machine" is fine, as long as it is the
    # frontend's port (see Settings.frontend_port).
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        resp = await preflight(client, origin)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3001",
        "http://localhost:8000",
        "http://localhost",
        "http://localhost:30000",
    ],
)
async def test_cors_rejects_other_ports(settings: Settings, origin: str) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        resp = await preflight(client, origin)
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


async def test_cors_port_is_configurable() -> None:
    settings = Settings(_env_file=None, frontend_port=3001)
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        assert (await preflight(client, "http://localhost:3001")).status_code == 200
        assert (await preflight(client, "http://localhost:3000")).status_code == 400


async def test_validation(settings: Settings) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        assert (await client.post("/api/chat", json={"message": ""})).status_code == 422
        assert (await client.post("/api/chat", json={})).status_code == 422


async def test_health(settings: Settings) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        body = (await client.get("/health")).json()
    assert body["status"] == "ok" and body["db_ok"] is True
    assert body["llm_provider"] == "lmstudio"
    # the cache is reported even before it has served a hit
    assert set(body["cache"]) == {"hits", "misses", "entries", "hit_rate"}
    assert body["cache"]["hits"] == 0 and body["cache"]["entries"] > 0
    # gif_search is offered only with a GIPHY key; the others always.
    assert body["tools"] == ["calculator", "knowledge_search", "weather"]


async def test_health_answers_head_for_uptime_probes(settings: Settings) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        resp = await client.head("/health")
    assert resp.status_code == 200
    assert resp.content == b""


async def test_openapi_documents_the_stream_and_the_health_shape(settings: Settings) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        spec = (await client.get("/openapi.json")).json()
        assert (await client.get("/docs")).status_code == 200

    chat_200 = spec["paths"]["/api/chat"]["post"]["responses"]["200"]
    assert list(chat_200["content"]) == ["text/event-stream"]  # not a JSON body
    components = spec["components"]["schemas"]
    # Every event in the union must be a resolvable component (Swagger UI follows these).
    refs = [v["$ref"] for v in chat_200["content"]["text/event-stream"]["schema"]["oneOf"]]
    assert all(r.startswith("#/components/schemas/") for r in refs)
    event_types = {components[r.rsplit("/", 1)[1]]["properties"]["type"]["const"] for r in refs}
    assert components["DoneEvent"]["properties"]["elapsed_ms"]
    assert event_types == {
        "conversation",
        "text_delta",
        "tool_start",
        "tool_result",
        "tool_error",
        "done",
        "error",
    }
    assert "404" in spec["paths"]["/api/chat"]["post"]["responses"]
    assert spec["components"]["schemas"]["HealthOut"]["properties"]["tools"]["type"] == "array"
    assert "locale" in spec["components"]["schemas"]["ChatRequest"]["properties"]
    assert {t["name"] for t in spec["tags"]} == {"chat", "health"}
    summaries = {
        (m.upper(), path): op["summary"]
        for path, ops in spec["paths"].items()
        for m, op in ops.items()
    }
    assert summaries[("PATCH", "/api/conversations/{conversation_id}")] == "Rename a conversation"
    assert "stream" in summaries[("POST", "/api/chat")].lower()


async def test_gif_tool_is_offered_when_a_key_is_configured() -> None:
    settings = Settings(_env_file=None, giphy_api_key="k")
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        body = (await client.get("/health")).json()
    assert body["tools"] == ["calculator", "knowledge_search", "weather", "gif_search"]


async def test_root_points_at_docs_instead_of_404(settings: Settings) -> None:
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"service": "personal-ai-assistant", "docs": "/docs", "health": "/health"}


async def test_sse_heartbeat_while_producer_is_quiet() -> None:
    async def slow_events() -> AsyncIterator[BaseModel]:
        yield TextDeltaEvent(text="a")
        await asyncio.sleep(0.12)
        yield TextDeltaEvent(text="b")

    frames = [f async for f in sse_stream(slow_events(), heartbeat_s=0.05)]
    assert frames[0] == 'event: text_delta\ndata: {"type":"text_delta","text":"a"}\n\n'
    assert ": ping\n\n" in frames[1:-1]
    assert frames[-1].endswith('"text":"b"}\n\n')


async def test_sse_propagates_producer_error() -> None:
    async def broken() -> AsyncIterator[BaseModel]:
        yield TextDeltaEvent(text="a")
        raise RuntimeError("producer died")

    with pytest.raises(RuntimeError, match="producer died"):
        [f async for f in sse_stream(broken(), heartbeat_s=1)]


async def test_sse_consumer_cancel_stops_producer() -> None:
    closed = asyncio.Event()

    async def forever() -> AsyncIterator[BaseModel]:
        try:
            while True:
                yield TextDeltaEvent(text="x")
                await asyncio.sleep(0.01)
        finally:
            closed.set()

    gen = sse_stream(forever(), heartbeat_s=1)
    await anext(gen)
    await gen.aclose()
    await asyncio.wait_for(closed.wait(), 1)


async def test_cors_allows_listed_origins_and_nothing_near_them(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"cors_origins": "https://app.example.com/, https://b.example"}
    )
    app, client = await make_client(settings, ScriptedChatModel(turns=[]))
    async with app.router.lifespan_context(app), client:
        ok = await preflight(client, "https://app.example.com")
        other = await preflight(client, "https://evil.example.com")
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://app.example.com"
    assert other.status_code == 400
