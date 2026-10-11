"""HTTP helper for tools that call public APIs. Every failure becomes a `ToolError` with a
message that is safe to show (no URLs or keys).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.tools.base import ToolError

DEFAULT_TIMEOUT_S = 5.0


def make_client(timeout_s: float = DEFAULT_TIMEOUT_S) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        headers={"User-Agent": "personal-ai-assistant/0.1"},
        follow_redirects=True,
    )


async def get_json(
    client: httpx.AsyncClient, url: str, params: dict[str, Any], *, service: str
) -> Any:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise ToolError(f"{service} did not answer in time") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise ToolError(f"{service} rejected the request (check the API key)") from exc
        if status == 429:
            raise ToolError(f"{service} is rate limiting us; try again in a moment") from exc
        raise ToolError(f"{service} returned an error ({status})") from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"{service} is unreachable") from exc
    except ValueError as exc:  # not JSON
        raise ToolError(f"{service} returned an unreadable reply") from exc
