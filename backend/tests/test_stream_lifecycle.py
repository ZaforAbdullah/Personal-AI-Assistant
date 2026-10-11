"""Streaming lifecycle: a client that leaves must not leave anything running or growing.

The chain is sse_stream -> ChatService -> stream_agent -> LangGraph -> model stream. Every
link must close the one below it the moment the consumer goes away (the model is billed per
token), and nothing may accumulate across requests.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
from collections.abc import AsyncIterator

from pydantic import BaseModel

from app.agent.events import TextDeltaEvent
from app.agent.service import ChatService
from app.db.repositories import InMemoryConversationRepository
from app.sse import sse_stream
from app.tools.calculator import CalculatorTool
from tests.conftest import WatchedModel


async def test_sse_producer_is_closed_promptly_when_the_consumer_stops() -> None:
    closed = asyncio.Event()

    async def producer() -> AsyncIterator[BaseModel]:
        try:
            while True:
                yield TextDeltaEvent(text="x")
                await asyncio.sleep(0)
        finally:
            closed.set()

    gen = sse_stream(producer(), heartbeat_s=5)
    await anext(gen)
    await gen.aclose()
    # No gc, no sleeping: the producer must already be closed, and its pump task gone.
    assert closed.is_set()
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []


async def test_sse_applies_backpressure_to_a_slow_consumer() -> None:
    produced = 0
    consumed = 0

    async def producer() -> AsyncIterator[BaseModel]:
        nonlocal produced
        for _ in range(2000):
            produced += 1
            yield TextDeltaEvent(text="x")
            await asyncio.sleep(0)

    gen = sse_stream(producer(), heartbeat_s=5)
    max_ahead = 0
    async for _ in gen:
        consumed += 1
        await asyncio.sleep(0.0005)  # a slow reader
        max_ahead = max(max_ahead, produced - consumed)
        if consumed == 300:
            break
    await gen.aclose()
    assert max_ahead <= 150, f"producer ran {max_ahead} events ahead of a slow client"


async def test_upstream_model_stream_is_closed_when_the_client_disconnects() -> None:
    WatchedModel.closed = False
    svc = ChatService(
        repo=InMemoryConversationRepository(),
        model=WatchedModel(),
        tools=[CalculatorTool()],
        max_tool_rounds=2,
        tool_timeout_s=1,
        history_limit=10,
    )
    events = svc.stream(None, "hi")
    seen = 0
    async for event in events:
        seen += event.type == "text_delta"
        if seen == 3:
            break  # the browser's Stop button
    await events.aclose()
    await asyncio.sleep(0)
    # the model must stop now, not whenever the GC gets to it
    assert WatchedModel.closed, "upstream stream still open after the consumer left"


async def test_no_async_generators_or_tasks_linger_after_many_disconnects() -> None:
    def live_asyncgens() -> int:
        gc.collect()
        return sum(1 for o in gc.get_objects() if inspect.isasyncgen(o))

    baseline_gens = live_asyncgens()
    baseline_tasks = len(asyncio.all_tasks())
    for _ in range(30):
        svc = ChatService(
            repo=InMemoryConversationRepository(),
            model=WatchedModel(),
            tools=[CalculatorTool()],
            max_tool_rounds=2,
            tool_timeout_s=1,
            history_limit=10,
        )
        events = sse_stream(svc.stream(None, "hi"), heartbeat_s=5)
        await anext(events)
        await anext(events)
        await events.aclose()
    del events, svc  # the loop variables would keep the last request's objects alive
    # let the loop run the last request's completion callbacks; a real leak would scale
    # with the 30 requests
    for _ in range(3):
        await asyncio.sleep(0)
    assert live_asyncgens() <= baseline_gens
    assert len(asyncio.all_tasks()) == baseline_tasks
