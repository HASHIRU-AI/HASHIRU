"""A long-lived MCP client connection the hub holds to another MCP server: a created tool's process or a
third-party server (stdio or streamable HTTP). Start it, notice if it dies, stop it."""
import asyncio
import contextlib
import logging
import os
import shutil
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import PaginatedRequestParams

__all__ = ["HttpTarget", "LinkError", "McpLink", "StderrTail"]

log = logging.getLogger(__name__)


class LinkError(RuntimeError):
    """An MCP server failed to start or to answer."""


@dataclass(frozen=True)
class HttpTarget:
    url: str
    headers: tuple[tuple[str, str], ...] = ()


class StderrTail(asyncio.Protocol):
    """The last `limit` bytes a process wrote to stderr, read from a pipe so nothing grows on disk or in memory."""

    def __init__(self, limit: int = 64_000) -> None:
        self.limit = limit
        self._buffer = bytearray()
        self._eof = asyncio.Event()
        read_fd, write_fd = os.pipe()
        self.writer = os.fdopen(write_fd, "w")  # given to the process as its stderr
        self._reader = os.fdopen(read_fd, "rb", buffering=0)
        self._transport: asyncio.BaseTransport | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.connect_read_pipe(lambda: self, self._reader)

    def data_received(self, data: bytes) -> None:
        self._buffer += data
        if len(self._buffer) > self.limit:
            del self._buffer[: len(self._buffer) - self.limit]

    def eof_received(self) -> bool:
        self._eof.set()
        return False

    def connection_lost(self, exc: Exception | None) -> None:
        self._eof.set()

    def text(self, limit: int = 1500) -> str:
        return bytes(self._buffer).decode("utf-8", errors="replace").strip()[-limit:]

    async def close(self) -> None:
        """Close our end of the pipe and collect what the (stopped) process still had in flight."""
        if not self.writer.closed:
            self.writer.close()
        if self._transport is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._eof.wait(), 1)
            self._transport.close()
        elif not self._reader.closed:
            self._reader.close()


async def list_all_tools(session: ClientSession, max_pages: int = 10) -> list[Any]:
    """Every tool a server lists, following pagination (a never-ending cursor stops after max_pages)."""
    tools: list[Any] = []
    cursor: str | None = None
    for _ in range(max_pages):
        page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
        tools += page.tools
        cursor = page.next_cursor
        if not cursor:
            break
    return tools


class McpLink:
    """Owns one connection. Its transport and session contexts are entered and left in one task (_serve)."""

    ping_every = 30.0  # seconds between liveness pings while idle
    ping_timeout = 10.0

    def __init__(self, name: str, target: StdioServerParameters | HttpTarget,
                 on_exit: Callable[["McpLink"], None], workdir: str | None = None) -> None:
        self.name = name
        self.target = target
        self.workdir = workdir
        self.stderr = StderrTail()
        self.session: ClientSession | None = None
        self.tools: list[Any] = []
        self._on_exit = on_exit
        self._stop = asyncio.Event()
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._task: asyncio.Task[None] | None = None

    async def start(self, timeout: float) -> None:
        """Connect and wait until the server has listed its tools. On any failure, everything is cleaned up."""
        try:
            await self.stderr.start()
            self._task = asyncio.create_task(self._serve())
            await asyncio.wait_for(asyncio.shield(self._ready), timeout)
        except BaseException:
            await self.close()
            raise

    @contextlib.asynccontextmanager
    async def _connect(self) -> AsyncIterator[tuple[Any, Any]]:
        if isinstance(self.target, StdioServerParameters):
            async with stdio_client(self.target, errlog=self.stderr.writer) as (read, write):
                yield read, write
        else:
            async with (httpx2.AsyncClient(headers=dict(self.target.headers),
                                           timeout=httpx2.Timeout(30, read=300)) as http,
                        streamable_http_client(self.target.url, http_client=http) as (read, write, *_)):
                yield read, write

    async def _serve(self) -> None:
        try:
            async with self._connect() as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                self.tools = await list_all_tools(session)
                self.session = session
                self._ready.set_result(None)
                await self._watch(session)
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                log.warning("MCP server %s stopped: %s", self.name, exc)
        finally:
            if not self._ready.done():
                self._ready.set_exception(LinkError("the server ended before it was ready"))
            if self.session is not None and not self._stop.is_set():
                self._on_exit(self)  # it was serving and nobody asked it to stop

    async def _watch(self, session: ClientSession) -> None:
        """Wait to be stopped, pinging while idle so a server that dies between calls is noticed."""
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.ping_every)
            except TimeoutError:
                await asyncio.wait_for(session.send_ping(), self.ping_timeout)  # raises if the server is gone

    async def close(self) -> None:
        """Disconnect (stopping a local process) and remove its scratch directory. Cancellation still propagates."""
        self._stop.set()
        task = self._task
        try:
            if task is not None and not task.done():
                if not self._ready.done():
                    task.cancel()  # not serving yet, so it would never see the stop event
                try:
                    await asyncio.wait_for(asyncio.shield(task), 10)
                except TimeoutError:
                    task.cancel()
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise  # our caller was cancelled; the task we cancelled ending is expected
        finally:
            if self._ready.done() and not self._ready.cancelled():
                self._ready.exception()  # mark it retrieved: start() may have stopped waiting already
            await self.stderr.close()
            if self.workdir:
                shutil.rmtree(self.workdir, ignore_errors=True)
