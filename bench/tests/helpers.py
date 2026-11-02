"""Builders and servers for PURSE's tests."""
import asyncio
import contextlib
import importlib.util
from pathlib import Path
from types import ModuleType

import uvicorn

from purse.jobs import Job


def job(uid: str, family: str = "arithmetic", level: int = 1, hidden: str = "low", value: int = 1, answer: str = "1",
        id: str = "") -> Job:
    return Job(id=id, uid=uid, family=family, level=level, hidden=hidden, value=value, prompt=f"task {uid}",
               answer=answer, features=())


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def example(name: str) -> ModuleType:
    """One of HASHIRU's examples, loaded from its file (examples/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoLLM:
    """For a hub whose agents should never think."""

    async def chat(self, model, messages, tools=None):
        raise AssertionError("no LLM call expected")


@contextlib.asynccontextmanager
async def served(app, port: int):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           timeout_graceful_shutdown=1))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        await task
