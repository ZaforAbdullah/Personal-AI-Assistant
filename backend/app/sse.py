"""Server-Sent Events encoding, with keep-alive comments while the producer is quiet."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing

from pydantic import BaseModel

_HEARTBEAT = ": ping\n\n"
_END = object()

# how far the producer may run ahead of the client; a stalled client stops the model
# call instead of growing the buffer
DEFAULT_BUFFER = 64


def format_sse(event: BaseModel) -> str:
    name = getattr(event, "type", "message")
    return f"event: {name}\ndata: {event.model_dump_json()}\n\n"


def _consume_result(task: asyncio.Task[None]) -> None:
    """Mark the pump's outcome as retrieved so an abandoned wait cannot log a warning."""
    if not task.cancelled():
        task.exception()


async def sse_stream(
    events: AsyncGenerator[BaseModel],
    *,
    heartbeat_s: float = 15.0,
    buffer: int = DEFAULT_BUFFER,
) -> AsyncIterator[str]:
    """Encode `events` as SSE frames, with a comment line whenever nothing was sent for
    `heartbeat_s` (tool calls and slow first tokens can look like a dead connection).

    The producer runs in its own task, so a heartbeat timeout never cancels it, behind a bounded
    queue, so a slow client applies backpressure. However this generator ends, the producer is
    cancelled and awaited, and its chain (service, graph, model stream) is closed at once with
    `aclosing`.
    """
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=buffer)

    async def pump() -> None:
        try:
            async with aclosing(events) as stream:
                async for event in stream:
                    await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # forwarded to the consumer, which re-raises it
            await queue.put(exc)
        else:
            await queue.put(_END)

    task = asyncio.create_task(pump())
    task.add_done_callback(_consume_result)
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
            except TimeoutError:
                yield _HEARTBEAT
                continue
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, BaseModel)
            yield format_sse(item)
    finally:
        task.cancel()
        # asyncio.wait, not gather: under Starlette's cancel scope gather would forward our own
        # cancellation to the pump as a second cancel(), interrupting LangGraph's cleanup and
        # leaving the model generating for nobody. wait() never cancels what it waits on.
        await asyncio.wait({task})
