"""Shared fakes: no API keys, no Ollama and no tokens spent."""
import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from hashiru.config import Settings
from hashiru.hub import Hub
from hashiru.session import Session


class FakeOllama:
    """Records loads and unloads. Every local model is 100 MB; 'tiny' reports 50 MB in sizes()."""

    def __init__(self, delay: float = 0.0, fail: Exception | None = None) -> None:
        self.delay, self.fail = delay, fail
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.on_unload: Any = None  # async hook called before an unload is recorded

    async def load(self, model: str) -> int:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        self.loaded.append(model)
        return 100

    async def unload(self, model: str) -> None:
        if self.on_unload:
            await self.on_unload(model)
        self.unloaded.append(model)

    async def sizes(self) -> dict[str, int]:
        return {"tiny": 50}


def make_settings(tmp_path: Any, **overrides: Any) -> Settings:
    base: dict[str, Any] = {"ceo_model": "gpt-4.1-mini", "models": ("gpt-4.1-mini", "ollama_chat/tiny"),
                            "data_dir": str(tmp_path / "data"), "trace_dir": str(tmp_path / "traces")}
    return Settings(**(base | overrides))


def ctx(session: Session, name: str) -> SimpleNamespace:
    """What an MCP tool sees when `name` calls it: the bearer token in the request headers."""
    return SimpleNamespace(headers={"authorization": f"Bearer {session.tokens[name]}"})


@pytest.fixture
def ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def hub(tmp_path: Any, ollama: FakeOllama) -> Hub:
    return Hub(make_settings(tmp_path), ollama=ollama)  # type: ignore[arg-type]


@pytest.fixture
def session(hub: Hub) -> Session:
    return hub.new_session("t", trace=False)
