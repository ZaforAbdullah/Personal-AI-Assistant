"""Scripted chat model for tests.

Plays back one turn per model call: a list of chunks to stream, or an exception to raise
mid-stream. It is a real `BaseChatModel`, so the whole graph runs offline, tool calls included.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from app.llm.errors import LLMError

Turn = list[Any]  # AIMessageChunk | Exception


class ScriptedChatModel(BaseChatModel):
    turns: list[Any] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)  # what the graph sent us

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _next_turn(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> Turn:
        self.calls.append({"messages": list(messages), "tools": kwargs.get("tools")})
        if not self.turns:
            raise LLMError("ScriptedChatModel has no more scripted turns")
        return self.turns.pop(0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return generate_from_stream(self._stream(messages, stop, run_manager, **kwargs))

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for item in self._next_turn(messages, kwargs):
            if isinstance(item, Exception):
                raise item
            yield ChatGenerationChunk(message=item)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for item in self._next_turn(messages, kwargs):
            if isinstance(item, Exception):
                raise item
            yield ChatGenerationChunk(message=item)


def text_turn(text: str, chunk_size: int = 4) -> Turn:
    """A plain streamed answer, split into small deltas."""
    return [
        AIMessageChunk(content=text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)
    ]


def tool_turn(name: str, arguments: str, call_id: str = "call_1", prefix: str = "") -> Turn:
    """A turn where the model asks for one tool, with the JSON arguments streamed in halves."""
    half = len(arguments) // 2
    turn: Turn = [AIMessageChunk(content=prefix)] if prefix else []
    turn += [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": name, "args": arguments[:half], "id": call_id, "index": 0}],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": arguments[half:], "id": None, "index": 0}],
        ),
    ]
    return turn


def tool_call_chunk(name: str, arguments: str, call_id: str, index: int) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": arguments, "id": call_id, "index": index}],
    )
