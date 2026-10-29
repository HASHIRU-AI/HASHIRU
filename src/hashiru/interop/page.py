"""The chat page: static files from hashiru/ui, served with headers that keep it self-contained.

The page loads nothing from other sites and talks only to this hub's /v1 API, so its Content-Security-Policy allows
only 'self'. frame-ancestors comes from settings (a Hugging Face Space shows the page in a frame).
"""
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["UI_DIR", "page_routes", "security_headers"]

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def security_headers(frame_ancestors: str) -> dict[str, str]:
    return {
        "content-security-policy": (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
            f"object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors {frame_ancestors}"),
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "cache-control": "no-cache",
    }


class WithHeaders:
    """Adds fixed headers to every response of the wrapped app."""

    def __init__(self, app: ASGIApp, headers: dict[str, str]) -> None:
        self.app = app
        self.headers = [(k.encode(), v.encode()) for k, v in headers.items()]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *self.headers]}
            await send(message)
        await self.app(scope, receive, send_with)


def page_routes(hub: "Hub") -> list[BaseRoute]:
    headers = security_headers(hub.settings.frame_ancestors)

    async def index(request: Request) -> Response:
        return FileResponse(UI_DIR / "index.html", headers=headers)

    return [Route("/", index, methods=["GET", "HEAD"]),
            Mount("/ui", app=WithHeaders(StaticFiles(directory=UI_DIR), headers))]
