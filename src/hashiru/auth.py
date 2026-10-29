"""Bearer-token gate in front of the hub. Everything not listed here is denied.

Internal routes, with tokens the hub issues per session:
- /mcp: any live caller the hub issued a token to (the user, the CEO, a manager). The caller becomes the
  request's ``scope["user"]``, so the MCP SDK only lets the credential that opened an MCP session use it.
- /s/{session}/agents/{name}/...: only the agent's hirer (or the user, for the CEO), in that session.

Public routes (hashiru.interop):
- /, /ui/..., /.well-known/agent-card.json: open (static files and the CEO's card).
- /v1/..., /a2a, /public/mcp: the api_key setting as a bearer token. Without an api_key the hub listens only on
  loopback, and these answer only requests addressed to a loopback name from this hub's own page (or no page), so a
  website open in the user's browser, or a page served on another local port, cannot reach them by DNS rebinding or
  a cross-site request.
"""
import hmac
from collections.abc import Mapping
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from hashiru.interop.openai_api import error

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["AuthGate", "is_open", "is_public", "token_of"]

LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}


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


def is_open(path: str) -> bool:
    """The chat page's files and the public agent card: no credentials needed."""
    return path in ("/", "/.well-known/agent-card.json") or path.startswith("/ui/")


def is_public(path: str) -> bool:
    """The public API routes, guarded by the api_key setting."""
    return path.startswith("/v1/") or path in ("/a2a", "/public/mcp") or path.startswith(("/a2a/", "/public/mcp/"))


def header(headers: list[tuple[bytes, bytes]], name: bytes) -> str:
    return next((v.decode("latin-1") for k, v in headers if k.lower() == name), "")


def loopback(netloc: str) -> bool:
    try:
        return (urlsplit(f"//{netloc}").hostname or "") in LOOPBACK_NAMES
    except ValueError:  # e.g. a malformed IPv6 literal
        return False


class AuthGate:
    def __init__(self, app: ASGIApp, hub: "Hub") -> None:
        self.app = app
        self.hub = hub

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # nothing here speaks websocket
            return
        if scope["type"] == "http" and is_open(scope["path"]):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http" and is_public(scope["path"]):
            refused = self.check_public(scope.get("headers", []))
            if refused:
                await error(*refused)(scope, receive, send)
                return
            await self.app(scope, receive, send)
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

    def check_public(self, headers: list[tuple[bytes, bytes]]) -> tuple[int, str] | None:
        """None if a public API request may proceed, else (status, reason)."""
        key = self.hub.settings.api_key
        if key:
            token = token_of(headers)
            if token is None or not hmac.compare_digest(token.encode(), key.encode()):
                return 401, "missing or wrong API key (send it as a bearer token)"
            return None
        host = header(headers, b"host")
        if not loopback(host):
            return 403, "without an api_key this hub only answers requests addressed to 127.0.0.1 or localhost"
        origin = header(headers, b"origin")
        if origin and origin.lower() != f"http://{host}".lower():  # only this hub's own page, not another port's
            return 403, "cross-site requests are refused without an api_key"
        return None

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
