"""Third-party MCP servers from the config, offered on the hub's /mcp as ``<server>__<tool>``.

The hub holds one MCP client connection per server (a local process over stdio, or a remote server over
streamable HTTP) and re-exports its tools, so agents see one tool list and every call still passes the
hub's auth gate, trace and result caps. A stdio server gets MCP's minimal environment plus only the
variables its config lists.
"""
import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from mcp import StdioServerParameters
from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED

from hashiru.config import McpServer
from hashiru.mcplink import HttpTarget, McpLink
from hashiru.toolsmith.checks import check_published, to_result
from hashiru.toolsmith.proxy import register_proxy

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["Gateway", "hub_name"]

log = logging.getLogger(__name__)
MAX_DESCRIPTION_CHARS = 1_000  # read by every manager on every turn
MAX_TOOLS_PER_SERVER = 64


def hub_name(server: str, tool: str) -> str:
    """Tool names in function calls allow only [a-zA-Z0-9_-], at most 64 characters."""
    return f"{server}__{re.sub(r'[^a-zA-Z0-9_-]', '_', tool)}"[:64]


class Gateway:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub
        self.links: dict[str, McpLink] = {}
        self.routes: dict[str, tuple[McpServer, str]] = {}  # hub tool name -> (server, its own tool name)
        self._lock = asyncio.Lock()
        self._reaping: set[asyncio.Task[None]] = set()

    def names(self) -> set[str]:
        return set(self.routes)

    def labels(self) -> dict[str, list[str]]:
        return {name: list(server.labels) for name, (server, _) in self.routes.items()}

    async def start(self) -> None:
        """Connect to every configured server at once. One that fails is logged and skipped, never fatal."""
        async with self._lock:
            servers = self.hub.settings.mcp_servers
            links = await asyncio.gather(*(self._open(s) for s in servers), return_exceptions=True)
            for server, link in zip(servers, links, strict=True):
                if isinstance(link, BaseException):
                    log.warning("skipping MCP server %s: %s", server.name, link)
                    continue
                self.links[server.name] = link
                try:
                    await self._offer(server, link)
                except Exception as exc:
                    log.warning("skipping MCP server %s: %s", server.name, exc)
                    await self._disconnect(server.name)

    async def stop(self) -> None:
        async with self._lock:
            for name in list(self.links):
                await self._disconnect(name)

    async def call(self, ctx: Context, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.hub.tools.who(ctx)
        route = self.routes.get(name)
        link = self.links.get(route[0].name) if route else None
        if route is None or link is None or link.session is None:
            return {"error": f"tool {name!r} is not available"}
        try:
            result = await link.session.call_tool(route[1], args,
                                                  read_timeout_seconds=self.hub.settings.tool_call_timeout)
        except MCPError as exc:
            if exc.error.code == CONNECTION_CLOSED:
                self._died(link)
            return {"error": f"{name} failed: {exc}"}
        except (OSError, RuntimeError) as exc:
            return {"error": f"{name} failed: {exc}"}
        return to_result(result)

    async def _open(self, server: McpServer) -> McpLink:
        target: StdioServerParameters | HttpTarget = (
            HttpTarget(server.url, server.headers) if server.url
            else StdioServerParameters(command=server.command, args=list(server.args), env=dict(server.env)))
        link = McpLink(server.name, target, self._died)
        await link.start(self.hub.settings.mcp_start_timeout)
        return link

    async def _offer(self, server: McpServer, link: McpLink) -> None:
        """Re-export the server's tools. A tool the hub cannot offer, or that is too large for every manager's
        prompt, is skipped on its own; a server cannot add more than MAX_TOOLS_PER_SERVER."""
        taken = {t.name for t in await self.hub.mcp.list_tools()} | {"ask_agent"}
        offered: list[str] = []
        for tool in link.tools:
            if len(offered) == MAX_TOOLS_PER_SERVER:
                log.warning("MCP server %s: offering only its first %d tools", server.name, MAX_TOOLS_PER_SERVER)
                break
            name = hub_name(server.name, tool.name)
            description, schema = (tool.description or "")[:MAX_DESCRIPTION_CHARS], dict(tool.input_schema)
            refused = (f"the name {name} is taken" if name in taken else check_published(description, schema))
            if not refused:
                try:
                    register_proxy(self.hub.mcp, name, description, schema, self.call)
                except Exception as exc:  # one odd schema must not take the rest of the server down
                    refused = str(exc)
            if refused:
                log.warning("MCP server %s: skipping %s: %s", server.name, tool.name, refused)
                continue
            self.routes[name] = (server, tool.name)
            taken.add(name)
            offered.append(name)
        log.info("MCP server %s offers %s", server.name, offered)

    async def _disconnect(self, server_name: str) -> None:
        link = self.links.pop(server_name)
        for name in [n for n, (s, _) in self.routes.items() if s.name == server_name]:
            del self.routes[name]
            self.hub.mcp.remove_tool(name)
        await link.close()

    def _died(self, link: McpLink) -> None:
        if self.links.get(link.name) is link:
            task = asyncio.get_running_loop().create_task(self._reap(link))
            self._reaping.add(task)
            task.add_done_callback(self._reaping.discard)

    async def _reap(self, link: McpLink) -> None:
        async with self._lock:
            if self.links.get(link.name) is link:
                log.warning("MCP server %s disconnected; its tools are unavailable until the hub restarts", link.name)
                await self._disconnect(link.name)
