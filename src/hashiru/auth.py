"""Bearer-token gate in front of the hub. Everything not listed here is denied.

- /mcp: any live caller the hub issued a token to (the user, the CEO, a manager). The caller becomes the
  request's ``scope["user"]``, so the MCP SDK only lets the credential that opened an MCP session use it.
- /s/{session}/agents/{name}/...: only the agent's hirer (or the user, for the CEO), in that session.
"""
from collections.abc import Mapping
from typing import TYPE_CHECKING

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["AuthGate", "token_of"]


def token_of(headers: Mapping[str, str] | list[tuple[bytes, bytes]]) -> str | None:
    """Bearer token from a header mapping (lower-case keys) or an ASGI header list."""
    if isinstance(headers, list):
        value = next((v.decode() for k, v in headers if k.lower() == b"authorization"), "")
    else:
        value = headers.get("authorization", "")
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def is_mcp(path: str) -> bool:
    return path == "/mcp" or path.startswith("/mcp/")


class AuthGate:
    def __init__(self, app: ASGIApp, hub: "Hub") -> None:
        self.app = app
        self.hub = hub

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # nothing here speaks websocket
            return
        if scope["type"] == "http":
            token = token_of(scope.get("headers", []))
            denied = self.check(scope["path"], token)
            if denied:
                status, message = denied
                await JSONResponse({"error": message}, status_code=status)(scope, receive, send)
                return
            who = self.hub.identity(token) if token else None
            if who and is_mcp(scope["path"]):
                principal = f"{who[0].id}/{who[1]}"
                scope = {**scope, "user": AuthenticatedUser(
                    AccessToken(token=token or "", client_id=principal, scopes=[], subject=principal))}
        await self.app(scope, receive, send)

    def check(self, path: str, token: str | None) -> tuple[int, str] | None:
        """None if the request may proceed, else (status, reason)."""
        if not (is_mcp(path) or path.startswith("/s/")):
            return 404, "not found"
        who = self.hub.identity(token) if token else None
        if who is None:
            return 401, "missing or unknown bearer token"
        if path.startswith("/s/"):
            parts = path.split("/")  # ['', 's', session, 'agents', name, ...]
            session, me = who
            if len(parts) < 5 or parts[3] != "agents" or parts[2] != session.id or not session.may_ask(me, parts[4]):
                return 403, "only an agent's hirer may call it"
        return None
