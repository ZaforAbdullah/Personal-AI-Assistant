"""gif_search tool: one G-rated GIF from GIPHY. Only registered when GIPHY_API_KEY is set, so
the model is never offered a tool that can't work.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import AsyncTool, ToolError
from app.tools.http import get_json

GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"


class GifArgs(BaseModel):
    query: str = Field(
        description="What the GIF should show, in a few words, e.g. 'happy cat' or 'high five'.",
        min_length=1,
        max_length=100,
    )


class GifSearchTool(AsyncTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "gif_search"
    description: str = (
        "Find a GIF (family-friendly) that matches a short description. Use it when the user "
        "asks for a gif, a reaction image, or something fun to look at. The GIF is shown to "
        "the user automatically; just describe it briefly in your reply."
    )
    args_schema: type[BaseModel] = GifArgs

    api_key: str = Field(exclude=True)
    client: httpx.AsyncClient = Field(exclude=True)

    async def _arun(self, query: str) -> dict[str, Any]:
        data = await get_json(
            self.client,
            GIPHY_SEARCH_URL,
            {"api_key": self.api_key, "q": query, "limit": 5, "rating": "g", "lang": "en"},
            service="GIPHY",
        )
        items = data.get("data") if isinstance(data, dict) else None
        if not items:
            raise ToolError(f"no GIF found for {query!r}")
        gif = items[0]
        image = (gif.get("images") or {}).get("fixed_height") or {}
        if not image.get("url"):
            raise ToolError("GIPHY returned a result without an image")
        return {
            "query": query,
            "title": gif.get("title") or query,
            "url": image["url"],
            "width": int(image.get("width") or 0),
            "height": int(image.get("height") or 0),
            "page": gif.get("url"),
            "source": "GIPHY",
        }
