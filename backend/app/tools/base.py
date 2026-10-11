"""Tool contract: LangChain `BaseTool`s, so `args_schema` is both the schema the model sees and
the validator for what it sends back. Tools raise `ToolError` for expected failures; anything
else is treated as a bug and reported generically.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool


class ToolError(Exception):
    """A tool could not complete; the message is safe to show to the model and the user."""


class AsyncTool(BaseTool):
    """Base for our async-only tools (the graph always awaits `_arun`)."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called
        raise NotImplementedError(f"{self.name} is async-only; use ainvoke")
