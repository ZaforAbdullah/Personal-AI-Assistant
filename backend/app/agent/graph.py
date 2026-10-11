"""The agent as a LangGraph state graph.

    START -> agent -> (tool calls?) -> tools -> agent -> ... -> END

`agent` streams the model, with tools bound until the round budget runs out so the last round
has to be prose. `tools` runs each requested tool under a timeout and reports progress on
LangGraph's custom stream channel. Tool failures become tool messages the model sees and error
events the user sees; they never abort the graph.

`stream_agent` turns the graph's stream into AgentEvents and records what was produced, so
ChatService can persist it, partial output included.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from app.agent.events import (
    AgentEvent,
    ErrorEvent,
    TextDeltaEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from app.tools.base import ToolError

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rounds: int  # model calls so far; bounds how many times tools may be used


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_tool_rounds: int = 4,
    tool_timeout_s: float = 15.0,
) -> CompiledStateGraph:
    by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(list(tools)) if tools else model

    async def call_model(state: AgentState) -> dict[str, Any]:
        rounds = state.get("rounds", 0)
        llm = model_with_tools if rounds < max_tool_rounds else model
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "rounds": rounds + 1}

    async def run_tools(state: AgentState) -> dict[str, Any]:
        emit = get_stream_writer()
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        results: list[ToolMessage] = []

        for i, bad in enumerate(last.invalid_tool_calls):
            call_id, name = bad.get("id") or f"invalid_{i}", bad.get("name") or "unknown"
            emit(ToolStartEvent(call_id=call_id, name=name, args={"_raw": bad.get("args") or ""}))
            error = f"could not parse tool arguments as JSON: {bad.get('error') or 'invalid'}"
            emit(ToolErrorEvent(call_id=call_id, name=name, error=error))
            results.append(_tool_message(call_id, name, {"error": error}))

        for i, call in enumerate(last.tool_calls):
            call_id, name, args = call.get("id") or f"call_{i}", call["name"], call["args"]
            emit(ToolStartEvent(call_id=call_id, name=name, args=args))
            error: str | None = None
            result: Any = None
            try:
                tool = by_name.get(name)
                if tool is None:
                    available = ", ".join(by_name) or "none"
                    raise ToolError(f"unknown tool {name!r}; available: {available}")
                result = await asyncio.wait_for(tool.ainvoke(args), timeout=tool_timeout_s)
            except ToolError as exc:
                error = str(exc)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}"
                    for e in exc.errors()
                )
                error = f"invalid arguments for {name}: {problems}"
            except TimeoutError:
                error = f"{name} timed out after {tool_timeout_s:g}s"
            except Exception:  # a bug inside a tool must not kill the stream
                log.exception("tool %s crashed", name)
                error = f"{name} failed unexpectedly"

            if error is None:
                emit(ToolResultEvent(call_id=call_id, name=name, result=result))
                results.append(_tool_message(call_id, name, result))
            else:
                emit(ToolErrorEvent(call_id=call_id, name=name, error=error))
                results.append(_tool_message(call_id, name, {"error": error}))
        return {"messages": results}

    def route(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and (last.tool_calls or last.invalid_tool_calls):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


def _tool_message(call_id: str, name: str, payload: Any) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, default=str),
        tool_call_id=call_id,
        name=name,
    )


@dataclass
class AgentRun:
    """What one run of the graph produced, for persistence."""

    produced: list[BaseMessage] = field(default_factory=list)  # completed node outputs
    partial_text: str = ""  # text of an agent turn that has not completed
    failed: bool = False

    def messages(self) -> list[BaseMessage]:
        out = list(self.produced)
        if self.partial_text:
            out.append(AIMessage(content=self.partial_text))
        return out


async def stream_agent(
    graph: CompiledStateGraph,
    messages: list[BaseMessage],
    run: AgentRun,
    *,
    recursion_limit: int = 25,
) -> AsyncIterator[AgentEvent]:
    """Run the graph once and yield UI events; fills `run` as nodes complete."""
    stream = graph.astream(
        {"messages": messages, "rounds": 0},
        config={"recursion_limit": recursion_limit},
        stream_mode=["messages", "custom", "updates"],
    )
    try:
        # aclosing: an early exit (Stop, closed tab) cancels the node and the model stream under it
        async with aclosing(stream) as chunks:
            async for mode, payload in chunks:
                if mode == "messages":
                    chunk, meta = payload
                    if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "agent":
                        if chunk.text:
                            run.partial_text += chunk.text
                            yield TextDeltaEvent(text=chunk.text)
                elif mode == "custom":
                    yield payload
                elif mode == "updates":
                    for node, update in payload.items():
                        if update:
                            run.produced.extend(update.get("messages", []))
                        if node == "agent":
                            run.partial_text = ""  # the full message is now in `produced`
    except GraphRecursionError:
        run.failed = True
        yield ErrorEvent(message="The assistant could not finish its reply.")
    except Exception as exc:
        log.warning("agent run failed: %s: %s", type(exc).__name__, exc)
        run.failed = True
        yield ErrorEvent(
            message=f"The model is unavailable right now ({type(exc).__name__}: {exc})."
        )
