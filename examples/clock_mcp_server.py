# /// script
# requires-python = ">=3.13"
# dependencies = ["mcp==2.2.0", "tzdata==2026.6"]  # tzdata: time zones where the system has none (Windows)
# ///
"""An MCP server that tells agents the date and time, which a model cannot know on its own.

HASHIRU starts it as a child process and talks to it over stdio; its tools reach agents as clock__now and
clock__days_between. Wire it up in a config file (examples/hashiru.yaml does):

    mcp_servers:
      clock: {command: uv, args: [run, examples/clock_mcp_server.py], labels: []}

(labels: [] because its results are computed here, not read from outside, so the trace need not mark them untrusted.)

Any MCP host can use it the same way, e.g. `claude mcp add clock -- uv run examples/clock_mcp_server.py`.
"""
from datetime import UTC, date, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("clock")


def zone(name: str) -> tzinfo:
    if name.upper() == "UTC":
        return UTC  # needs no time zone database
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):  # ValueError: not a zone key at all, e.g. a path
        # A ToolError's message reaches the agent, which can correct its call; any other exception counts as a
        # crash, and the agent sees only "Error executing tool".
        raise ToolError(f"unknown time zone {name!r}; use an IANA name such as Europe/Paris") from None


@server.tool()
def now(timezone: str = "UTC") -> str:
    """The current date and time in an IANA time zone (e.g. America/Los_Angeles), as ISO 8601 with the weekday."""
    moment = datetime.now(zone(timezone))
    return f"{moment.isoformat(timespec='seconds')} ({moment:%A})"


@server.tool()
def days_between(start: str, end: str) -> int:
    """Days from start to end, both ISO dates (YYYY-MM-DD). Negative if end is earlier."""
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError as exc:
        raise ToolError(f"{exc}; use ISO dates such as 2026-10-29") from None


if __name__ == "__main__":
    server.run("stdio")
