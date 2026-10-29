"""Serve one agent-created tool over MCP stdio, in the tool's own uv environment (never in the hub).

Usage: python -I -u child.py <module.py> <name>

-u matters: while serving, the SDK points fd 1 at stderr, but a buffered print would be flushed onto the
MCP wire after it restores fd 1 at exit.

Only the standard library and mcp may be imported here: hashiru is not installed in this environment.
"""
import functools
import importlib.util
import inspect
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

MAX_RESULT_CHARS = 1_000_000  # the hub cuts results further; this keeps a runaway result off the wire


def checked(value: Any) -> Any:
    size = len(json.dumps(value, default=str))
    if size > MAX_RESULT_CHARS:
        raise ToolError(f"the result has {size} characters; return at most {MAX_RESULT_CHARS}")
    return value


def reporting(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Pass the tool's own exceptions to the agent (as ToolError), so it can fix its code.

    Without this the SDK reports only "Error executing tool <name>".
    """
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            try:
                return checked(await fn(*args, **kwargs))
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        return run_async

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return checked(fn(*args, **kwargs))
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    return run


def import_quietly(path: str, name: str) -> Any:
    """Import the tool's module with fd 1 (the MCP wire) pointed at stderr, so nothing it writes, from print or
    from C code, reaches the wire before the server starts."""
    spec = importlib.util.spec_from_file_location(f"hashiru_tool_{name}", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    wire = os.dup(1)
    os.dup2(2, 1)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout.flush()
        os.dup2(wire, 1)
        os.close(wire)
    return module


def main() -> None:
    path, name = sys.argv[1], sys.argv[2]
    fn = getattr(import_quietly(path, name), name, None)
    if not callable(fn):
        sys.exit(f"the code must define a function named {name}")
    if not (fn.__doc__ or "").strip():
        sys.exit(f"{name} needs a docstring: it becomes the tool's description")
    server = MCPServer(f"hashiru-tool-{name}")
    server.add_tool(reporting(fn), name=name)
    server.run("stdio")  # while serving, the SDK points fd 1 at stderr, so prints in the tool are safe too


if __name__ == "__main__":
    main()
