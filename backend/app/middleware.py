"""Request id and timing for every request.

Pure ASGI, because BaseHTTPMiddleware buffers streaming responses and would break SSE.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

log = logging.getLogger("app.http")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = b"x-request-id"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_id = headers.get(REQUEST_ID_HEADER, b"").decode() or uuid.uuid4().hex[:12]
        scope["state"] = {**scope.get("state", {}), "request_id": request_id}
        started = time.perf_counter()
        method, path = scope.get("method", "?"), scope.get("path", "?")

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message.get("headers", []),
                    (REQUEST_ID_HEADER, request_id.encode()),
                ]
                if path != "/health":  # keep probes out of the log
                    log.info(
                        "%s %s -> %s in %dms rid=%s",
                        method,
                        path,
                        message["status"],
                        (time.perf_counter() - started) * 1000,
                        request_id,
                    )
            await send(message)

        await self.app(scope, receive, send_with_id)
