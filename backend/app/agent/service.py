"""ChatService: persistence around the LangGraph agent.

Creates or loads the conversation, stores the user message, builds the transcript, runs the
graph and stores what it produced, including partial output when the model fails or the client
leaves.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from typing import Any
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.agent.events import AgentEvent, ConversationEvent, DoneEvent, ToolStartEvent
from app.agent.graph import AgentRun, build_graph, stream_agent
from app.agent.prompts import system_prompt
from app.db.repositories import ConversationRepository, MessageRecord, MessageStatus

log = logging.getLogger(__name__)


class ConversationNotFound(Exception):
    pass


def _log_write_failure(future: asyncio.Future[Any]) -> None:
    if not future.cancelled() and (exc := future.exception()) is not None:
        log.error("could not store a partial reply", exc_info=exc)


class ChatService:
    def __init__(
        self,
        *,
        repo: ConversationRepository,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        max_tool_rounds: int,
        tool_timeout_s: float,
        history_limit: int,
    ) -> None:
        self._repo = repo
        self._history_limit = history_limit
        self._graph = build_graph(
            model, tools, max_tool_rounds=max_tool_rounds, tool_timeout_s=tool_timeout_s
        )
        # two nodes per round plus slack, as a hard stop if routing misbehaves
        self._recursion_limit = 2 * (max_tool_rounds + 1) + 2
        # in-flight turns and partial-reply writes, drained at shutdown before the pool closes
        self._pending: set[asyncio.Future[Any]] = set()

    def _track(self, future: asyncio.Future[Any]) -> asyncio.Future[Any]:
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)
        return future

    async def aclose(self, wait_s: float = 5.0) -> None:
        """Wait for in-flight turns and their partial-reply writes.

        Runs at shutdown before the pool is disposed: uvicorn cancels open requests and starts the
        shutdown without waiting for them.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while self._pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                log.warning(
                    "shutdown: %d turn(s) still unfinished after %gs", len(self._pending), wait_s
                )
                return
            await asyncio.wait(set(self._pending), timeout=remaining)

    async def stream(
        self, conversation_id: UUID | None, text: str, *, locale: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        turn = self._track(asyncio.get_running_loop().create_future())
        try:
            # aclosing: leaving early closes _stream now, so the partial reply is stored promptly
            async with aclosing(self._stream(conversation_id, text, locale)) as events:
                async for event in events:
                    yield event
        finally:
            if not turn.done():
                turn.set_result(None)

    async def _stream(
        self, conversation_id: UUID | None, text: str, locale: str | None
    ) -> AsyncIterator[AgentEvent]:
        if conversation_id is None:
            conv = await self._repo.create_conversation()
        else:
            found = await self._repo.get_conversation(conversation_id)
            if found is None:
                raise ConversationNotFound(str(conversation_id))
            conv = found
        yield ConversationEvent(conversation_id=conv.id)

        await self._repo.add_messages([MessageRecord(conv.id, "user", text)])
        history = await self._repo.list_messages(conv.id, last=self._history_limit)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt(locale)),
            *history_to_messages(history),
        ]

        run = AgentRun()
        started = time.perf_counter()
        tool_calls = 0
        try:
            # close the agent stream (and the model call under it) before storing the partial reply
            async with aclosing(
                stream_agent(self._graph, messages, run, recursion_limit=self._recursion_limit)
            ) as agent_events:
                async for event in agent_events:
                    if isinstance(event, ToolStartEvent):
                        tool_calls += 1
                    yield event
        except BaseException:
            # client left or something broke: keep what we have so history stays coherent
            log.info("stream for %s interrupted; persisting partial reply", conv.id)
            # separate tracked task: under uvicorn the request's cancel scope stays cancelled, so an
            # await here can be abandoned; aclose() waits for the task at shutdown
            write = self._track(
                asyncio.ensure_future(self._persist(conv.id, run.messages(), "interrupted"))
            )
            write.add_done_callback(_log_write_failure)
            try:
                await asyncio.shield(write)
            except Exception:  # a storage error must not mask why we are here
                pass  # already logged by the callback
            raise

        saved = await self._persist(conv.id, run.messages(), "error" if run.failed else "complete")
        last_assistant = next((m for m in reversed(saved) if m.role == "assistant"), None)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "turn conv=%s elapsed=%dms tools=%d status=%s",
            conv.id,
            elapsed_ms,
            tool_calls,
            "error" if run.failed else "complete",
        )
        yield DoneEvent(
            message_id=last_assistant.id if last_assistant else None,
            elapsed_ms=elapsed_ms,
            tool_calls=tool_calls,
        )

    async def _persist(
        self, conversation_id: UUID, produced: list[BaseMessage], status: MessageStatus
    ) -> list[MessageRecord]:
        records = [r for m in produced if (r := message_to_record(m, conversation_id, status))]
        if records:
            await self._repo.add_messages(records)
        return records


def message_to_record(
    m: BaseMessage, conversation_id: UUID, status: MessageStatus
) -> MessageRecord | None:
    if isinstance(m, AIMessage):
        calls: list[dict[str, Any]] = [
            {"id": tc["id"], "name": tc["name"], "arguments": json.dumps(tc["args"])}
            for tc in m.tool_calls
        ] + [
            {"id": bad.get("id"), "name": bad.get("name"), "arguments": bad.get("args") or ""}
            for bad in m.invalid_tool_calls
        ]
        if not m.text and not calls:
            return None  # an empty turn (e.g. the model failed before saying anything)
        return MessageRecord(conversation_id, "assistant", m.text, calls or None, status=status)
    if isinstance(m, ToolMessage):
        return MessageRecord(
            conversation_id, "tool", m.text, tool_call_id=m.tool_call_id, tool_name=m.name
        )
    if isinstance(m, HumanMessage):
        return MessageRecord(conversation_id, "user", m.text)
    return None


def history_to_messages(history: list[MessageRecord]) -> list[BaseMessage]:
    """Rebuild a transcript the model API will accept.

    Drops tool calls with no stored result and tool results with no call (the window can cut
    through an exchange, and an interrupted turn may never have stored its results). Prose is
    always kept.
    """
    answered = {m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id}
    requested = {tc["id"] for m in history if m.tool_calls for tc in m.tool_calls}

    out: list[BaseMessage] = []
    for m in history:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            calls = [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": _parse_args(tc["arguments"]),
                    "type": "tool_call",
                }
                for tc in (m.tool_calls or [])
                if tc["id"] in answered
            ]
            if m.content or calls:
                out.append(AIMessage(content=m.content, tool_calls=calls))
        elif m.role == "tool" and m.tool_call_id in requested:
            out.append(
                ToolMessage(content=m.content, tool_call_id=m.tool_call_id, name=m.tool_name)
            )
    # The first non-system message must be from the user.
    while out and not isinstance(out[0], HumanMessage):
        out.pop(0)
    return out


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
