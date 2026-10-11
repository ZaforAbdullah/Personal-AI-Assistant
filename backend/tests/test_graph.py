"""Agent graph behaviour, driven end to end by a scripted chat model."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool

from app.llm.errors import LLMError
from app.llm.fake import ScriptedChatModel, text_turn, tool_call_chunk, tool_turn
from tests.conftest import joined_text, run_graph, types


async def test_text_only_reply_streams_deltas(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(turns=[text_turn("Hello there!")])
    events, run = await run_graph(model, tools)

    assert set(types(events)) == {"text_delta"}
    assert joined_text(events) == "Hello there!"
    assert [type(m) for m in run.produced] == [AIMessage]
    assert run.produced[0].text == "Hello there!"
    assert run.partial_text == "" and run.failed is False
    assert model.calls[0]["tools"]  # tools were offered
    assert [t["function"]["name"] for t in model.calls[0]["tools"]] == [t.name for t in tools]


async def test_tool_call_then_answer(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[
            tool_turn("calculator", '{"expression": "17 * 23 + 5"}', call_id="c1"),
            text_turn("That is 396."),
        ]
    )
    events, run = await run_graph(model, tools, "what is 17*23+5?")

    assert types(events)[:2] == ["tool_start", "tool_result"]
    assert events[0].args == {"expression": "17 * 23 + 5"}
    assert events[1].result == {"expression": "17 * 23 + 5", "result": 396}
    assert joined_text(events) == "That is 396."

    # The second model call saw the tool call and its result.
    second = model.calls[1]["messages"]
    assert isinstance(second[-2], AIMessage) and second[-2].tool_calls[0]["id"] == "c1"
    assert isinstance(second[-1], ToolMessage) and second[-1].tool_call_id == "c1"
    assert '"result": 396' in second[-1].text
    assert [type(m) for m in run.produced] == [AIMessage, ToolMessage, AIMessage]


async def test_text_before_tool_call_is_streamed(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[tool_turn("echo", '{"text": "x"}', prefix="Let me check. "), text_turn("Done.")]
    )
    events, _ = await run_graph(model, tools)
    assert types(events) == ["text_delta", "tool_start", "tool_result", "text_delta", "text_delta"]


async def test_tool_error_is_reported_and_fed_back(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[tool_turn("failing", '{"text": "x"}'), text_turn("Sorry, no luck.")]
    )
    events, _ = await run_graph(model, tools)

    assert types(events)[:2] == ["tool_start", "tool_error"]
    assert events[1].error == "nothing found for that"
    assert '"error"' in model.calls[1]["messages"][-1].text
    assert joined_text(events) == "Sorry, no luck."


async def test_tool_crash_does_not_leak_details(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(turns=[tool_turn("crashing", '{"text": "x"}'), text_turn("ok")])
    events, _ = await run_graph(model, tools)
    assert events[1].type == "tool_error"
    assert "boom" not in events[1].error
    assert events[1].error == "crashing failed unexpectedly"


async def test_tool_timeout(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(turns=[tool_turn("slow", '{"text": "x"}'), text_turn("timed out")])
    events, _ = await run_graph(model, tools, tool_timeout_s=0.05)
    assert events[1].type == "tool_error"
    assert "timed out" in events[1].error


async def test_unknown_tool_and_bad_args(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[
            [
                tool_call_chunk("ghost", "{}", "a", 0),
                tool_call_chunk("echo", '{"nope": 1}', "b", 1),
            ],
            text_turn("recovered"),
        ]
    )
    events, _ = await run_graph(model, tools)
    errors = [e for e in events if e.type == "tool_error"]
    assert [e.name for e in errors] == ["ghost", "echo"]
    assert "unknown tool" in errors[0].error
    assert "invalid arguments for echo: text" in errors[1].error
    assert joined_text(events) == "recovered"


async def test_truncated_json_arguments_become_invalid_args(tools: list[BaseTool]) -> None:
    # LangChain repairs truncated JSON to what it can ({}), so validation catches it.
    model = ScriptedChatModel(
        turns=[[tool_call_chunk("echo", '{"text": ', "a", 0)], text_turn("ok")]
    )
    events, run = await run_graph(model, tools)
    assert types(events)[:2] == ["tool_start", "tool_error"]
    assert "invalid arguments for echo" in events[1].error
    assert joined_text(events) == "ok"
    assert isinstance(run.produced[1], ToolMessage)  # the model was told about it


async def test_non_json_arguments_are_reported(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[[tool_call_chunk("echo", "not json at all", "a", 0)], text_turn("ok")]
    )
    events, run = await run_graph(model, tools)
    assert types(events)[:2] == ["tool_start", "tool_error"]
    assert events[0].args == {"_raw": "not json at all"}
    assert "could not parse" in events[1].error
    assert joined_text(events) == "ok"
    assert isinstance(run.produced[1], ToolMessage)


async def test_max_rounds_forces_prose_answer(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[
            tool_turn("echo", '{"text": "1"}'),
            tool_turn("echo", '{"text": "2"}'),
            text_turn("final"),
        ]
    )
    events, _ = await run_graph(model, tools, max_tool_rounds=2)
    assert joined_text(events) == "final"
    assert model.calls[0]["tools"] and model.calls[1]["tools"]
    assert model.calls[2]["tools"] is None  # last round: no tools bound


async def test_llm_failure_mid_stream_yields_error_and_keeps_partial(
    tools: list[BaseTool],
) -> None:
    model = ScriptedChatModel(
        turns=[[AIMessageChunk(content="Partial "), LLMError("connection reset")]]
    )
    events, run = await run_graph(model, tools)
    assert types(events) == ["text_delta", "error"]
    assert "unavailable" in events[-1].message
    assert run.failed is True
    assert run.partial_text == "Partial "
    assert [m.text for m in run.messages()] == ["Partial "]


async def test_tool_call_fragments_are_reassembled(tools: list[BaseTool]) -> None:
    model = ScriptedChatModel(
        turns=[
            [
                tool_call_chunk("calculator", "", "c9", 0),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": None, "args": '{"expr', "id": None, "index": 0}],
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": None, "args": 'ession": "2', "id": None, "index": 0}
                    ],
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": None, "args": ' + 2"}', "id": None, "index": 0}],
                ),
            ],
            text_turn("4"),
        ]
    )
    events, _ = await run_graph(model, tools)
    assert events[0].call_id == "c9" and events[0].args == {"expression": "2 + 2"}
    assert events[1].result["result"] == 4
