"""Shared fixtures. Nothing here touches the network or a database."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.agent.events import AgentEvent
from app.agent.graph import AgentRun, build_graph, stream_agent
from app.llm.fake import ScriptedChatModel
from app.tools.base import AsyncTool, ToolError
from app.tools.calculator import CalculatorTool


class EchoArgs(BaseModel):
    text: str = Field(description="text to echo")


class EchoTool(AsyncTool):
    name: str = "echo"
    description: str = "Echo the text back."
    args_schema: type[BaseModel] = EchoArgs

    async def _arun(self, text: str) -> dict[str, Any]:
        return {"echo": text}


class FailingTool(AsyncTool):
    name: str = "failing"
    description: str = "Always raises ToolError."
    args_schema: type[BaseModel] = EchoArgs

    async def _arun(self, text: str) -> Any:
        raise ToolError("nothing found for that")


class CrashingTool(AsyncTool):
    name: str = "crashing"
    description: str = "Raises an unexpected exception (a bug)."
    args_schema: type[BaseModel] = EchoArgs

    async def _arun(self, text: str) -> Any:
        raise RuntimeError("boom")


class SlowTool(AsyncTool):
    name: str = "slow"
    description: str = "Takes forever."
    args_schema: type[BaseModel] = EchoArgs

    async def _arun(self, text: str) -> Any:
        await asyncio.sleep(60)
        return {"never": True}


@pytest.fixture
def tools() -> list[BaseTool]:
    return [CalculatorTool(), EchoTool(), FailingTool(), CrashingTool(), SlowTool()]


def transcript(user: str = "hello") -> list[BaseMessage]:
    return [SystemMessage(content="sys"), HumanMessage(content=user)]


async def run_graph(
    model: ScriptedChatModel,
    tools: list[BaseTool],
    user: str = "hello",
    **graph_kwargs: Any,
) -> tuple[list[Any], AgentRun]:
    """Build a graph around the scripted model, run one turn, return (events, run)."""
    graph = build_graph(model, tools, **graph_kwargs)
    run = AgentRun()
    events = [e async for e in stream_agent(graph, transcript(user), run)]
    return events, run


async def collect(events: AsyncIterator[AgentEvent]) -> list[Any]:
    return [e async for e in events]


def types(events: list[Any]) -> list[str]:
    return [e.type for e in events]


def joined_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if e.type == "text_delta")


class WatchedModel(ScriptedChatModel):
    """A model that streams forever and records when its stream is closed."""

    closed: bool = False

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        try:
            while True:
                yield ChatGenerationChunk(message=AIMessageChunk(content="tok "))
                await asyncio.sleep(0.001)
        finally:
            type(self).closed = True
