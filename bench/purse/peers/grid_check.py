"""PURSE's grid_check tool, an MCP server over stdio: whether a proposed solution satisfies every clue of a logic-grid
puzzle. Checking is cheaper than solving (as unit tests are), so it is free, but it answers at most MAX_CALLS times
per puzzle so it cannot be used to search.

    python -m purse.peers.grid_check   (HASHIRU starts it from mcp_servers; PYTHONPATH must include bench/)
"""
import hashlib
import json

from mcp.server.mcpserver import MCPServer

from purse.families.grids import parse, satisfies

__all__ = ["MAX_CALLS", "check", "server"]

MAX_CALLS = 3
server = MCPServer("purse")
_counts: dict[str, int] = {}


def check(puzzle: str, answer: str, counts: dict[str, int]) -> str:
    try:
        p = parse(puzzle)
    except ValueError as exc:
        return f"invalid: {exc}"
    key = hashlib.sha256(puzzle.encode()).hexdigest()
    if counts.get(key, 0) >= MAX_CALLS:
        return "limit reached"
    counts[key] = counts.get(key, 0) + 1
    start, end = answer.find("{"), answer.rfind("}")
    try:
        assignment = json.loads(answer[start:end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        return "invalid"
    return "valid" if isinstance(assignment, dict) and satisfies(p, assignment) else "invalid"


@server.tool()
def grid_check(puzzle: str, answer: str) -> str:
    """Check a logic-grid solution: give the puzzle's full text and the solution as JSON ({"1": {"color": ...}, ...}).
    Answers valid or invalid, at most 3 times per puzzle."""
    return check(puzzle, answer, _counts)


if __name__ == "__main__":
    server.run("stdio")
