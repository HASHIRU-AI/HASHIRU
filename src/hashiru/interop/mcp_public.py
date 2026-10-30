"""HASHIRU as an MCP server for MCP hosts (Claude Code, Cursor, ...), at /public/mcp.

Tools: `ask` sends a message to the CEO (pass the `session` from an earlier answer to continue that conversation,
with its agents still hired) and `end_session` closes one. The agents' work arrives as progress notifications.
Resource: `hashiru://sessions/{session}`, a conversation's agents and pools.
"""
import contextlib
import json
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import BaseRoute

from hashiru.interop.activity import describe
from hashiru.interop.conversations import Busy

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["MCP_PATH", "PublicMCP"]

MCP_PATH = "/public/mcp"
INSTRUCTIONS = ("HASHIRU is a budget-aware multi-agent system. Use `ask` for tasks that benefit from a team of agents "
                "(research, multi-step work); its CEO hires agents on local and cloud models within a dollar budget. "
                "Keep the returned `session` to ask follow-ups, and call `end_session` when done.")


class PublicMCP:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub
        self.server = MCPServer("hashiru", instructions=INSTRUCTIONS)
        self.server.tool()(self.ask)
        self.server.tool()(self.end_session)
        self.server.resource("hashiru://sessions/{session}", name="session",
                             description="A HASHIRU conversation's agents, pools and spending.",
                             mime_type="application/json")(self.usage)
        # With a key, the gate authenticates and any Host may reach us (e.g. through a reverse proxy); without one,
        # the SDK's own loopback Host check stays on, alongside the gate's.
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False) if hub.settings.api_key else None
        app = self.server.streamable_http_app(streamable_http_path=MCP_PATH, host=hub.settings.host,
                                              transport_security=security)
        self.routes: list[BaseRoute] = list(app.routes)

    async def ask(self, ctx: Context, text: str, session: str = "", private: bool = False) -> dict[str, Any]:
        """Ask HASHIRU's CEO, which hires and directs agents within a budget, and wait for its answer.

        Pass the session from an earlier answer to continue that conversation; leave it empty to start a new one.
        private=true keeps this message and the rest of the conversation on local models.
        """
        convs = self.hub.conversations
        conv = convs.get(session) if session else None
        if session and conv is None:
            return {"error": f"no open session {session!r} (it may have expired); leave session empty to start anew"}
        if not text.strip():
            return {"error": "text is empty"}
        try:
            conv = conv or await convs.open("mcp")
            turn = convs.ask(conv, text, private=private)
        except Busy as exc:
            return {"error": str(exc)}
        step, answer = 0, {}
        async with contextlib.aclosing(turn) as events:
            async for item in events:
                if item["kind"] == "answer":
                    answer = item
                elif line := describe(item):
                    step += 1
                    await ctx.report_progress(step, None, line)
        return {"answer": answer["text"], "session": conv.id, "cost_usd": answer["cost_usd"]}

    async def end_session(self, session: str) -> dict[str, Any]:
        """Close a HASHIRU session: fire its agents and free their memory."""
        conv = self.hub.conversations.get(session)
        if conv is None:
            return {"error": f"no open session {session!r}"}
        await self.hub.conversations.close(conv)
        return {"closed": session}

    def usage(self, session: str) -> str:
        conv = self.hub.conversations.get(session)
        if conv is None:
            raise ValueError(f"no open session {session!r}")
        return json.dumps(self.hub.conversations.state(conv))
