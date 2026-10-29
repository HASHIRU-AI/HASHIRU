"""The bearer-token gate in front of /mcp and the agents' A2A endpoints."""
import pytest

from hashiru.auth import token_of
from hashiru.session import USER


@pytest.mark.parametrize(("headers", "token"), [
    ({"authorization": "Bearer abc"}, "abc"),
    ({"authorization": "bearer  abc "}, "abc"),
    ({"authorization": "Basic abc"}, None),
    ({"authorization": "Bearer "}, None),
    ({}, None),
    ([(b"Authorization", b"Bearer xyz")], "xyz"),
    ([(b"accept", b"*/*")], None),
])
def test_token_of(headers, token):
    assert token_of(headers) == token


def test_mcp_needs_a_known_token(hub, session):
    gate = hub.asgi
    assert gate.check("/mcp", None)[0] == 401
    assert gate.check("/mcp", "forged")[0] == 401
    assert gate.check("/mcp", session.tokens["ceo"]) is None


def test_only_the_hirer_reaches_an_agent(hub, session):
    session.ledger.add("w1", "worker", "cloud", "gpt-4.1-mini", "p", "ceo", 1, 0.1)
    gate, base = hub.asgi, f"/s/{session.id}/agents"
    assert gate.check(f"{base}/ceo/", session.tokens[USER]) is None
    assert gate.check(f"{base}/w1/.well-known/agent-card.json", session.tokens["ceo"]) is None
    assert gate.check(f"{base}/w1/", session.tokens[USER])[0] == 403      # the user talks only to the CEO
    assert gate.check(f"{base}/ceo/", session.tokens["ceo"])[0] == 403    # nobody hired the CEO but the user
    assert gate.check(f"{base}/nobody/", session.tokens["ceo"])[0] == 403
    assert gate.check(f"{base}/w1/", None)[0] == 401


def test_sessions_cannot_reach_each_other(hub, session):
    other = hub.new_session("other", trace=False)
    assert hub.asgi.check(f"/s/{session.id}/agents/ceo/", other.tokens[USER])[0] == 403


def test_closed_sessions_lose_their_tokens(hub, session):
    import asyncio
    token = session.tokens["ceo"]
    asyncio.run(hub.close_session(session))
    assert hub.identity(token) is None and hub.asgi.check("/mcp", token)[0] == 401


@pytest.mark.parametrize("path", ["/", "//s/x/agents/ceo/", "/web/x", "/S/x/agents/ceo/"])
def test_unknown_paths_are_denied_by_default(hub, session, path):
    assert hub.asgi.check(path, session.tokens[USER])[0] == 404


def test_gate_binds_mcp_sessions_to_the_caller(hub, session):
    import asyncio

    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    seen = {}

    async def app(scope, receive, send):
        seen["user"] = scope.get("user")
    hub.asgi.app = app
    headers = [(b"authorization", f"Bearer {session.tokens['ceo']}".encode())]
    asyncio.run(hub.asgi({"type": "http", "path": "/mcp", "headers": headers}, None, None))
    assert isinstance(seen["user"], AuthenticatedUser)
    assert seen["user"].access_token.subject == f"{session.id}/ceo"


def test_websockets_are_refused(hub):
    import asyncio
    sent = []

    async def send(message):
        sent.append(message)
    asyncio.run(hub.asgi({"type": "websocket", "path": "/mcp", "headers": []}, None, send))
    assert sent and sent[0]["type"] == "websocket.close"
