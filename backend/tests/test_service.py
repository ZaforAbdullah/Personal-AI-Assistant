"""ChatService: persistence around the graph, history windowing, interruption handling."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.service import ChatService, ConversationNotFound, history_to_messages
from app.db.repositories import InMemoryConversationRepository, MessageRecord
from app.llm.errors import LLMError
from app.llm.fake import ScriptedChatModel, text_turn, tool_turn
from tests.conftest import collect, joined_text, types


def make_service(repo, model, tools, **kw) -> ChatService:
    return ChatService(
        repo=repo,
        model=model,
        tools=tools,
        max_tool_rounds=kw.get("max_tool_rounds", 4),
        tool_timeout_s=kw.get("tool_timeout_s", 1.0),
        history_limit=kw.get("history_limit", 30),
    )


async def test_new_conversation_persists_full_exchange(tools: list[BaseTool]) -> None:
    repo = InMemoryConversationRepository()
    model = ScriptedChatModel(
        turns=[tool_turn("calculator", '{"expression": "2+2"}', call_id="k"), text_turn("4")]
    )
    svc = make_service(repo, model, tools)

    events = await collect(svc.stream(None, "2+2?"))
    assert types(events)[0] == "conversation"
    assert types(events)[-1] == "done"
    conv_id = events[0].conversation_id

    stored = await repo.list_messages(conv_id)
    assert [m.role for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1].tool_calls == [
        {"id": "k", "name": "calculator", "arguments": '{"expression": "2+2"}'}
    ]
    assert stored[2].tool_call_id == "k" and stored[2].tool_name == "calculator"
    assert stored[3].content == "4" and stored[3].status == "complete"
    assert events[-1].message_id == stored[3].id


async def test_follow_up_sees_history(tools: list[BaseTool]) -> None:
    repo = InMemoryConversationRepository()
    model = ScriptedChatModel(
        turns=[text_turn("I'm your assistant."), text_turn("You asked my name.")]
    )
    svc = make_service(repo, model, tools)

    first = await collect(svc.stream(None, "who are you?"))
    conv_id = first[0].conversation_id
    await collect(svc.stream(conv_id, "what did I ask?"))

    sent = model.calls[1]["messages"]
    assert [m.type for m in sent] == ["system", "human", "ai", "human"]
    assert sent[2].text == "I'm your assistant."


async def test_unknown_conversation_raises(tools: list[BaseTool]) -> None:
    svc = make_service(InMemoryConversationRepository(), ScriptedChatModel(), tools)
    with pytest.raises(ConversationNotFound):
        await collect(svc.stream(uuid4(), "hi"))


async def test_llm_error_persists_partial_with_error_status(tools: list[BaseTool]) -> None:
    repo = InMemoryConversationRepository()
    model = ScriptedChatModel(turns=[[AIMessageChunk(content="Half"), LLMError("down")]])
    svc = make_service(repo, model, tools)
    events = await collect(svc.stream(None, "hi"))

    assert types(events) == ["conversation", "text_delta", "error", "done"]
    stored = await repo.list_messages(events[0].conversation_id)
    assert stored[-1].role == "assistant"
    assert stored[-1].content == "Half" and stored[-1].status == "error"


async def test_client_disconnect_persists_partial_as_interrupted(tools: list[BaseTool]) -> None:
    repo = InMemoryConversationRepository()
    model = ScriptedChatModel(
        turns=[text_turn("This is a long reply that gets cut off", chunk_size=2)]
    )
    svc = make_service(repo, model, tools)

    conv_id: UUID | None = None
    gen = svc.stream(None, "hi")
    async for event in gen:
        if event.type == "conversation":
            conv_id = event.conversation_id
        if event.type == "text_delta":
            break  # simulate the consumer going away mid-stream
    await gen.aclose()

    assert conv_id is not None
    stored = await repo.list_messages(conv_id)
    assert stored[-1].role == "assistant"
    assert stored[-1].status == "interrupted"
    assert stored[-1].content == "Th"


async def test_cancellation_persists_partial_as_interrupted(tools: list[BaseTool]) -> None:
    repo = InMemoryConversationRepository()
    model = ScriptedChatModel(
        turns=[tool_turn("slow", '{"text": "x"}', prefix="Hold on. "), text_turn("late")]
    )
    svc = make_service(repo, model, tools, tool_timeout_s=30)

    seen = []

    async def consume() -> None:
        async for event in svc.stream(None, "hi"):
            seen.append(event)

    task = asyncio.create_task(consume())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(e.type == "tool_start" for e in seen):
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    conv_id = seen[0].conversation_id
    stored = await repo.list_messages(conv_id)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[1].status == "interrupted" and stored[1].content == "Hold on. "
    assert stored[1].tool_calls and stored[1].tool_calls[0]["name"] == "slow"


async def test_history_limit_and_transcript_repair() -> None:
    conv = uuid4()
    history = [
        MessageRecord(conv, "user", "q1"),
        MessageRecord(
            conv, "assistant", "", tool_calls=[{"id": "t1", "name": "echo", "arguments": "{}"}]
        ),
        MessageRecord(conv, "tool", '{"echo": ""}', tool_call_id="t1", tool_name="echo"),
        MessageRecord(conv, "assistant", "a1"),
        MessageRecord(conv, "user", "q2"),
        # interrupted turn: tool requested, never answered
        MessageRecord(
            conv,
            "assistant",
            "partial",
            tool_calls=[{"id": "t2", "name": "echo", "arguments": '{"text": "z"}'}],
            status="interrupted",
        ),
        MessageRecord(conv, "user", "q3"),
    ]
    # Window cuts between the tool call and its result: the orphan tool result is dropped and
    # so is the assistant answer before the first user message, so the API sees a valid start.
    t = history_to_messages(history[2:])
    assert [m.type for m in t] == ["human", "ai", "human"]
    assert t[0].text == "q2"

    full = history_to_messages(history)
    assert [m.type for m in full] == ["human", "ai", "tool", "ai", "human", "ai", "human"]
    assert isinstance(full[1], AIMessage) and full[1].tool_calls[0]["id"] == "t1"
    assert isinstance(full[2], ToolMessage) and full[2].tool_call_id == "t1"
    assert full[5].text == "partial" and full[5].tool_calls == []  # unanswered call stripped
    assert isinstance(full[6], HumanMessage)


async def test_empty_assistant_turn_is_not_stored(tools: list[BaseTool]) -> None:
    repo = InMemoryConversationRepository()
    svc = make_service(repo, ScriptedChatModel(turns=[[LLMError("nope")]]), tools)
    events = await collect(svc.stream(None, "hi"))
    stored = await repo.list_messages(events[0].conversation_id)
    assert [m.role for m in stored] == ["user"]
    assert joined_text(events) == ""
