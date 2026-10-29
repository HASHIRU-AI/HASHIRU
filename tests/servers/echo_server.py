"""A tiny third-party MCP server for the gateway tests (stdio)."""
import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool(name="shout.loud")  # some servers use names that function calls do not allow
def shout(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


@server.tool()
def env(name: str) -> str:
    """Read one of this server's environment variables."""
    return os.environ.get(name, "<unset>")


@server.tool()
def die() -> str:
    """Exit the process."""
    os._exit(1)


if __name__ == "__main__":
    server.run("stdio")
