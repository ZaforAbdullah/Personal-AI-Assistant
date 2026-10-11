"""Leak check: drives the real app for many rounds and watches what grows.

    cd backend && .venv/bin/python -m scripts.soak --rounds 3000

Each round does a tool turn, a follow-up, list and rename, a stream the client abandons, and
deletes. The model is scripted, so no keys or network. After warm-up it samples allocated
blocks, asyncio tasks, live async generators and cache size; a healthy app plateaus. Exits
non-zero if memory keeps climbing.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import inspect
import sys

import httpx
from langchain_core.messages import ToolMessage

from app.agent.service import ChatService
from app.config import Settings
from app.db.repositories import InMemoryConversationRepository, InMemoryKnowledgeRepository
from app.llm.embeddings import FakeEmbedder
from app.llm.fake import ScriptedChatModel, text_turn, tool_turn
from app.main import create_app
from app.sse import sse_stream

_DELIBERATE_LEAK: list[dict[str, str]] = []  # only ever filled by --simulate-leak


class EndlessModel(ScriptedChatModel):
    """A tool call, then an answer, forever. (ScriptedChatModel records every call for
    assertions, which would itself grow without bound here, so that is bypassed.)"""

    simulate_leak: bool = False

    def _next_turn(self, messages, kwargs):  # type: ignore[override]
        if self.simulate_leak:
            _DELIBERATE_LEAK.append({"turn": "x" * 200})  # what a real leak looks like
        if messages and isinstance(messages[-1], ToolMessage):
            return text_turn("The answer is 4.")
        return tool_turn("calculator", '{"expression": "2+2"}')


class SlowForever(ScriptedChatModel):
    """Never finishes a reply, and generates it lazily: for the turns the client abandons."""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        while True:
            yield ChatGenerationChunk(message=AIMessageChunk(content="word "))
            await asyncio.sleep(0)


def snapshot(app) -> dict[str, float]:
    gc.collect()
    state = app.state.container
    return {
        "blocks": sys.getallocatedblocks(),
        "tasks": len(asyncio.all_tasks()) - 1,
        "asyncgens": sum(1 for o in gc.get_objects() if inspect.isasyncgen(o)),
        "cache": len(state.cache) if state.cache is not None else 0,
    }


async def run(rounds: int, cache_entries: int, cache: bool, simulate_leak: bool = False) -> bool:
    settings = Settings(
        _env_file=None,
        cache_enabled=cache,
        cache_max_entries=cache_entries,
        sse_heartbeat_s=30,
    )
    repo = InMemoryConversationRepository()
    app = create_app(
        settings,
        chat_model=EndlessModel(simulate_leak=simulate_leak),
        embedder=FakeEmbedder(),
        conversations=repo,
        knowledge=InMemoryKnowledgeRepository(),
    )
    rows: list[tuple[int, dict[str, float]]] = []
    every = max(1, rounds // 6)
    async with app.router.lifespan_context(app):
        service = app.state.container.chat_service
        slow_service = ChatService(
            repo=app.state.container.conversations,
            model=SlowForever(),
            tools=[],
            max_tool_rounds=1,
            tool_timeout_s=1,
            history_limit=10,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            for i in range(1, rounds + 1):
                r = await c.post("/api/chat", json={"message": "what is 2+2?", "locale": "en"})
                cid = r.text.split('"conversation_id":"')[1].split('"')[0]
                await c.post("/api/chat", json={"message": "and again?", "conversation_id": cid})
                await c.get("/api/conversations")
                await c.get(f"/api/conversations/{cid}/messages")
                await c.patch(f"/api/conversations/{cid}", json={"title": f"chat {i}"})

                # a turn the client abandons after a few chunks (the Stop button)
                events = sse_stream(slow_service.stream(None, "stop me"), heartbeat_s=30)
                first = await anext(events)
                for _ in range(2):
                    await anext(events)
                await events.aclose()
                stopped_cid = first.split('"conversation_id":"')[1].split('"')[0]
                await slow_service.aclose()  # the partial reply is written by a background task

                # delete what the round created, so remaining growth is a leak and not data
                await c.delete(f"/api/conversations/{cid}")
                await c.delete(f"/api/conversations/{stopped_cid}")
                if i == 1 or i % every == 0:
                    await service.aclose()
                    await slow_service.aclose()
                    rows.append((i, snapshot(app)))
    print(f"\ncache={'on' if cache else 'off'} (max {cache_entries} entries), {rounds} rounds")
    print(f"{'round':>7} {'alloc blocks':>13} {'tasks':>6} {'asyncgens':>10} {'cache':>6}")
    for i, s in rows:
        print(
            f"{i:>7} {s['blocks']:>13.0f} {s['tasks']:>6.0f} {s['asyncgens']:>10.0f}"
            f" {s['cache']:>6.0f}"
        )
    # compare the second half of the run: a plateau moves by noise, a leak by a constant per round
    warm = rows[len(rows) // 2 :]
    growth = (warm[-1][1]["blocks"] - warm[0][1]["blocks"]) / max(1, warm[-1][0] - warm[0][0])
    tasks_ok = all(s["tasks"] <= rows[0][1]["tasks"] + 1 for _, s in rows)
    gens_ok = all(s["asyncgens"] <= rows[0][1]["asyncgens"] + 2 for _, s in rows)
    print(
        f"steady-state growth: {growth:.2f} allocated blocks/round | tasks stable: {tasks_ok} "
        f"| generators stable: {gens_ok}"
    )
    return growth < 2 and tasks_ok and gens_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--cache-entries", type=int, default=200)
    parser.add_argument(
        "--simulate-leak",
        action="store_true",
        help="self-test: leak on purpose and confirm this check fails (exit code 1)",
    )
    args = parser.parse_args()
    ok = True
    for cache in (True, False):
        ok &= asyncio.run(run(args.rounds, args.cache_entries, cache, args.simulate_leak))
    print("\nRESULT:", "no leak detected" if ok else "MEMORY KEEPS GROWING")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
