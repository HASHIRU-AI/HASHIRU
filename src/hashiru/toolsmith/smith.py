"""Agent-created tools: each one runs as its own MCP server process, re-exported on the hub's /mcp.

The process is ``uv run --no-project --with mcp --with <deps> python -I -u child.py``, so a tool's packages never
touch the hub's environment. It starts with MCP's minimal default environment (it does not inherit HASHIRU's
environment variables) in a scratch directory. This isolates dependencies, not the host: the code runs as the
user and can read their files and other processes' details, which is why tool creation is off unless
``tool_creation=true``. Created tools are shared by every session on the hub and persist in ``tools_dir``.
"""
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp import StdioServerParameters
from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED

from hashiru.mcplink import LinkError as ToolError
from hashiru.mcplink import McpLink
from hashiru.toolsmith.checks import check_entry, check_published, check_request, digest, to_result
from hashiru.toolsmith.proxy import register_proxy

if TYPE_CHECKING:
    from hashiru.hub import Hub
    from hashiru.session import Session

__all__ = ["Toolsmith"]

log = logging.getLogger(__name__)

CHILD = Path(__file__).with_name("child.py")
MANIFEST = "manifest.json"


class Toolsmith:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub
        self.dir = Path(hub.settings.tools_dir)
        self.procs: dict[str, McpLink] = {}
        self._lock = asyncio.Lock()
        self._closing = False
        self._reaping: set[asyncio.Task[None]] = set()

    def names(self) -> set[str]:
        return set(self.procs)

    # ---- lifecycle ----

    async def start(self) -> None:
        """Start every saved tool. Each is checked again as if newly created; a tool that fails is skipped."""
        async with self._lock:
            self._closing = False
            taken = await self._taken()
            for name, entry in self._manifest().items():
                try:
                    await self._load_saved(name, entry, taken)
                    taken.add(name)
                except Exception as exc:  # one bad tool must never stop the hub
                    log.warning("skipping saved tool %s: %s", name, exc)

    async def stop(self) -> None:
        async with self._lock:
            self._closing = True
            for name in list(self.procs):
                await self._retire(name)

    # ---- the tools agents call ----

    async def create(self, name: str, code: str, dependencies: list[str], creator: str) -> dict[str, Any]:
        async with self._lock:
            if self._closing:
                return {"error": "the hub is shutting down"}
            if refused := check_request(name, code, dependencies, await self._taken()):
                return {"error": refused}
            data = code.encode()
            try:
                proc = await self._start_tool(name, data, dependencies)
            except ToolError as exc:
                return {"error": f"{name} was not created: {exc}"}
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{name}.py").write_bytes(data)
            entry = {"dependencies": dependencies, "sha256": digest(data), "created_by": creator,
                     "created_at": round(time.time(), 3)}
            self._save_manifest(self._manifest() | {name: entry})
        return {"created": name, "sha256": entry["sha256"], "parameters": proc.tools[0].input_schema}

    async def delete(self, name: str, session: "Session") -> dict[str, Any]:
        """Only the session that created a tool may delete it, unless that session has ended."""
        async with self._lock:
            if name not in self.procs:
                return {"error": f"no created tool named {name!r}"}
            owner = str(self._manifest().get(name, {}).get("created_by", "")).split("/")[0]
            if owner != session.id and owner in self.hub.sessions:
                return {"error": f"{name} belongs to another session"}
            await self._retire(name)
            (self.dir / f"{name}.py").unlink(missing_ok=True)
            manifest = self._manifest()
            manifest.pop(name, None)
            self._save_manifest(manifest)
        return {"deleted": name}

    async def call(self, ctx: Context, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Forward one call from an agent (already authenticated by the gate) to the tool's process."""
        self.hub.tools.who(ctx)
        proc = self.procs.get(name)
        if proc is None or proc.session is None:
            return {"error": f"tool {name!r} is not running"}
        try:
            result = await proc.session.call_tool(name, args, read_timeout_seconds=self.hub.settings.tool_call_timeout)
        except MCPError as exc:
            if exc.error.code == CONNECTION_CLOSED:
                self._died(proc)
            return {"error": f"{name} failed: {exc}"}
        except (OSError, RuntimeError) as exc:
            return {"error": f"{name} failed: {exc}"}
        return to_result(result)

    # ---- processes ----

    def command(self, path: Path, name: str, dependencies: list[str], workdir: str) -> StdioServerParameters:
        """The one place that decides how a tool runs (a container backend would change only this)."""
        uv = shutil.which("uv")
        if uv is None:
            raise ToolError("uv is not installed, so created tools cannot run")
        args = ["run", "--no-project", "--quiet", "--python", f"{sys.version_info.major}.{sys.version_info.minor}",
                "--with", f"mcp=={version('mcp')}"]
        for dep in dependencies:
            args += ["--with", dep]
        args += ["python", "-I", "-u", str(CHILD), str(path), name]  # -u: see child.py
        return StdioServerParameters(command=uv, args=args, cwd=workdir)  # env=None: MCP's minimal default

    async def _start_tool(self, name: str, data: bytes, dependencies: list[str]) -> McpLink:
        """Run `data` (a private copy, so later edits to tools_dir cannot change what was checked), check what it
        serves, and re-export it. Any failure stops the process; the error carries the end of its stderr."""
        workdir = tempfile.mkdtemp(prefix=f"hashiru-tool-{name}-")
        try:
            path = Path(workdir) / f"{name}.py"
            path.write_bytes(data)
            proc = McpLink(name, self.command(path, name, dependencies, workdir), self._died, workdir)
        except Exception as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise exc if isinstance(exc, ToolError) else ToolError(f"{type(exc).__name__}: {exc}") from exc
        try:
            await proc.start(self.hub.settings.tool_start_timeout)
            if [t.name for t in proc.tools] != [name]:
                raise ToolError(f"expected exactly one tool named {name}, got {[t.name for t in proc.tools]}")
            tool = proc.tools[0]
            description, schema = tool.description or "", dict(tool.input_schema)
            if refused := check_published(description, schema):
                raise ToolError(refused)
            self._register(name, description, schema)
        except Exception as exc:
            await proc.close()
            reason = "the tool did not start in time" if isinstance(exc, TimeoutError) else str(exc)
            if not isinstance(exc, ToolError | TimeoutError):
                reason = f"{type(exc).__name__}: {exc}"
            tail = proc.stderr.text()
            raise ToolError(f"{reason}\n{tail}" if tail else reason) from None
        except BaseException:
            await proc.close()
            raise
        self.procs[name] = proc
        return proc

    async def _load_saved(self, name: str, entry: Any, taken: set[str]) -> None:
        if refused := check_entry(entry):
            raise ToolError(refused)
        data = (self.dir / f"{name}.py").read_bytes()
        if digest(data) != entry["sha256"]:
            raise ToolError("its code does not match the manifest")
        dependencies = entry.get("dependencies", [])
        if refused := check_request(name, data.decode(), dependencies, taken):
            raise ToolError(refused)
        await self._start_tool(name, data, dependencies)

    def _register(self, name: str, description: str, schema: dict[str, Any]) -> None:
        try:
            register_proxy(self.hub.mcp, name, description, schema, self.call)
        except ValueError as exc:
            raise ToolError(str(exc)) from None

    async def _retire(self, name: str) -> None:
        proc = self.procs.pop(name)
        self.hub.mcp.remove_tool(name)
        await proc.close()

    def _died(self, proc: McpLink) -> None:
        """A serving process ended on its own: unregister it (it comes back when the hub restarts)."""
        if self.procs.get(proc.name) is proc:
            task = asyncio.get_running_loop().create_task(self._reap(proc))
            self._reaping.add(task)
            task.add_done_callback(self._reaping.discard)

    async def _reap(self, proc: McpLink) -> None:
        async with self._lock:
            if self.procs.get(proc.name) is proc:
                log.warning("tool %s exited; it is unavailable until the hub restarts", proc.name)
                await self._retire(proc.name)

    async def _taken(self) -> set[str]:
        return {t.name for t in await self.hub.mcp.list_tools()} | {"ask_agent"}

    # ---- manifest ----

    def _manifest(self) -> dict[str, Any]:
        path = self.dir / MANIFEST
        try:
            manifest = json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable %s: %s", path, exc)
            return {}
        return manifest if isinstance(manifest, dict) else {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        tmp = self.dir / f".{MANIFEST}.tmp"
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, self.dir / MANIFEST)
